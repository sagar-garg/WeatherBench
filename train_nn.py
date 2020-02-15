from src.score import *
from src.data_generator import *
from src.networks import *
from src.utils import *
import os
import ast, re
import numpy as np
import xarray as xr
import tensorflow as tf
import tensorflow.keras as keras
from configargparse import ArgParser
import pickle
import pdb

class LRUpdate(object):
    def __init__(self, init_lr, step, divide):
        # From goo.gl/GXQaK6
        self.init_lr = init_lr
        self.step = step
        self.drop = 1. / divide

    def __call__(self, epoch):
        lr = self.init_lr * np.power(self.drop, np.floor((epoch) / self.step))
        print(f'Learning rate = {lr}')
        return lr

def main(datadir, var_dict, output_vars, filters, kernels, lr, activation, dr, batch_size, early_stopping_patience, epochs, exp_id,
         model_save_dir, pred_save_dir, train_years, valid_years, test_years, lead_time, gpu, iterative,
         norm_subsample, data_subsample, lr_step, lr_divide, network_type, restore_best_weights,
         negative_slope, bn_position, nt_in, dt_in, use_bias, l2):
    os.environ["CUDA_VISIBLE_DEVICES"]=str(gpu)
    # Limit TF memory usage
    limit_mem()

    # Open dataset and create data generators
    ds = xr.merge(
        [xr.open_mfdataset(f'{datadir}/{var}/*.nc', combine='by_coords')
         for var in var_dict.keys()],
        fill_value=0
    )

    ds_train = ds.sel(time=slice(*train_years))
    ds_valid = ds.sel(time=slice(*valid_years))
    ds_test = ds.sel(time=slice(*test_years))

    dg_train = DataGenerator(
        ds_train, var_dict, lead_time, batch_size=batch_size, output_vars=output_vars,
        data_subsample=data_subsample, norm_subsample=norm_subsample, nt_in=nt_in, dt_in=dt_in
    )
    dg_valid = DataGenerator(
        ds_valid, var_dict, lead_time, batch_size=batch_size, mean=dg_train.mean, std=dg_train.std,
        shuffle=False, output_vars=output_vars, nt_in=nt_in, dt_in=dt_in
    )
    dg_test =  DataGenerator(
        ds_test, var_dict, lead_time, batch_size=batch_size, mean=dg_train.mean, std=dg_train.std,
        shuffle=False, output_vars=output_vars, nt_in=nt_in, dt_in=dt_in
    )
    print(f'Mean = {dg_train.mean}; Std = {dg_train.std}')

    # Build model
    # TODO: Flexible input shapes and optimizer
    if network_type == 'fc':
        model = build_cnn(filters, kernels, input_shape=(32, 64, len(dg_train.data.level)*nt_in),
                          activation=activation, dr=dr, l2=l2)
    elif network_type == 'fc_no_periodic':
        model = build_cnn(filters, kernels, input_shape=(32, 64, len(dg_train.data.level)*nt_in), activation=activation,
                          dr=dr, periodic=False, l2=l2)
    elif network_type =='resnet':
        assert activation == 'relu', 'Resnet only with ReLU'
        model = build_resnet(filters, kernels, input_shape=(32, 64, len(dg_train.data.level)*nt_in),
                             negative_slope=negative_slope, bn_position=bn_position, use_bias=use_bias, l2=l2)
    model.compile(keras.optimizers.Adam(lr), 'mse')
    print(model.summary())

    # Learning rate settings
    callbacks = []
    if early_stopping_patience is not None:
        callbacks.append(tf.keras.callbacks.EarlyStopping(
                          monitor='val_loss',
                          min_delta=0,
                          patience=early_stopping_patience,
                          verbose=1,
                          mode='auto',
                          # restore_best_weights=restore_best_weights   # Only TF>1.12
                      ))
    if lr_step is not None:
        callbacks.append(keras.callbacks.LearningRateScheduler(
            LRUpdate(lr, lr_step, lr_divide)
        ))

    # Train model
    # TODO: Learning rate schedule
    history = model.fit_generator(dg_train, epochs=epochs, validation_data=dg_valid,
                      callbacks=callbacks
                      )
    print(f'Saving model weights: {model_save_dir}/{exp_id}.h5')
    model.save_weights(f'{model_save_dir}/{exp_id}.h5')
    print(f'Saving training_history: {model_save_dir}/{exp_id}_history.pkl')
    to_pickle(history.history, f'{model_save_dir}/{exp_id}_history.pkl')


    # Create predictions
    preds = create_predictions(model, dg_test)
    print(f'Saving predictions: {pred_save_dir}/{exp_id}.nc')
    preds.to_netcdf(f'{pred_save_dir}/{exp_id}.nc')

    # Print score in real units
    # TODO: Make flexible for other states
    z500_valid = load_test_data(f'{datadir}geopotential_500', 'z')
    t850_valid = load_test_data(f'{datadir}temperature_850', 't')
    try:
        print(compute_weighted_rmse(preds.z, z500_valid).load())
    except:
        print('Z500 not found in predictions.')
    try:
        print(compute_weighted_rmse(preds.t, t850_valid).load())
    except:
        print('T850 not found in predictions.')

if __name__ == '__main__':
    p = ArgParser()
    p.add_argument('-c', '--my-config', is_config_file=True, help='config file path')
    p.add_argument('--datadir', type=str, required=True, help='Path to data')
    p.add_argument('--exp_id', type=str, required=True, help='Experiment identifier')
    p.add_argument('--model_save_dir', type=str, required=True, help='Path to save model')
    p.add_argument('--pred_save_dir', type=str, required=True, help='Path to save predictions')
    p.add_argument('--var_dict', required=True, help='Variables: as an ugly dictionary...')
    p.add_argument('--output_vars', nargs='+', help='Output variables. Format {var}_{level}', default=None)
    p.add_argument('--filters', type=int, nargs='+', required=True, help='Filters for each layer')
    p.add_argument('--kernels', type=int, nargs='+', required=True, help='Kernel size for each layer')
    p.add_argument('--lead_time', type=int, required=True, help='Forecast lead time')
    p.add_argument('--iterative', type=bool, default=False, help='Is iterative forecast')
    p.add_argument('--iterative_max_lead_time', type=int, default=5*24, help='Max lead time for iterative forecasts')
    p.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    p.add_argument('--activation', type=str, default='elu', help='Activation function')
    p.add_argument('--dr', type=float, default=0, help='Dropout rate')
    p.add_argument('--batch_size', type=int, default=128, help='batch_size')
    p.add_argument('--epochs', type=int, default=100, help='epochs')
    p.add_argument('--early_stopping_patience', type=int, default=None, help='Early stopping patience')
    p.add_argument('--restore_best_weights', type=bool, default=True, help='ES parameter')
    p.add_argument('--train_years', type=str, nargs='+', default=('1979', '2015'), help='Start/stop years for training')
    p.add_argument('--valid_years', type=str, nargs='+', default=('2016', '2016'), help='Start/stop years for validation')
    p.add_argument('--test_years', type=str, nargs='+', default=('2017', '2018'), help='Start/stop years for testing')
    p.add_argument('--data_subsample', type=int, default=1, help='Subsampling for training data')
    p.add_argument('--norm_subsample', type=int, default=1, help='Subsampling for mean/std')
    p.add_argument('--gpu', type=int, default=0, help='Which GPU')
    p.add_argument('--lr_step', type=int, default=None, help='LR decay step')
    p.add_argument('--lr_divide', type=int, default=None, help='LR decay division factor')
    p.add_argument('--network_type', type=str, default='fc', help='Type')
    p.add_argument('--negative_slope', type=float, default=0, help='Slope for Resnet ReLU')
    p.add_argument('--bn_position', type=str, default=None, help='pre, mid or post')
    p.add_argument('--nt_in', type=int, default=1, help='Number of input time steps')
    p.add_argument('--dt_in', type=int, default=1, help='Time step of intput time steps (after subsampling)')
    p.add_argument('--use_bias', type=bool, default=True, help='Use bias in resnet convs')
    p.add_argument('--l2', type=float, default=0, help='Weight decay')
    args = p.parse_args()

    main(
        datadir=args.datadir,
        var_dict=ast.literal_eval(args.var_dict),
        output_vars=args.output_vars,
        filters=args.filters,
        kernels=args.kernels,
        lr=args.lr,
        activation=args.activation,
        dr=args.dr,
        batch_size=args.batch_size,
        epochs=args.epochs,
        early_stopping_patience=args.early_stopping_patience,
        exp_id=args.exp_id,
        model_save_dir=args.model_save_dir,
        pred_save_dir=args.pred_save_dir,
        train_years=args.train_years,
        valid_years=args.valid_years,
        test_years=args.test_years,
        lead_time=args.lead_time,
        gpu=args.gpu,
        iterative=args.iterative,
        data_subsample=args.data_subsample,
        norm_subsample=args.norm_subsample,
        lr_step=args.lr_step,
        lr_divide=args.lr_divide,
        network_type=args.network_type,
        restore_best_weights=args.restore_best_weights,
        negative_slope=args.negative_slope,
        bn_position=args.bn_position,
        nt_in=args.nt_in,
        dt_in=args.dt_in,
        use_bias=args.use_bias,
        l2=args.l2
    )
