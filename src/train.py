from src.score import *
from src.data_generator import *
from src.networks import *
from src.utils import *
from src.regrid import regrid
from src.clr import OneCycleLR
import os
import ast, re
import numpy as np
import xarray as xr
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.mixed_precision import experimental as mixed_precision
from configargparse import ArgParser


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

def load_data(var_dict, datadir, cmip, cmip_dir, train_years, valid_years, test_years,
              lead_time, batch_size, output_vars, data_subsample, norm_subsample,
              nt_in, dt_in, only_test=False, ext_mean=None, ext_std=None, cont_time=False,
              multi_dt=1, verbose=0,
              train_tfr_files=None, valid_tfr_files=None, test_tfr_files=None, tfr_num_parallel_calls=1,
              tfr_buffer_size=1000, tfr_prefetch=None, y_roll=None, X_roll=None, discard_first=None,
              min_lead_time=None, tp_log=None, tfr_out=False, tfr_out_idxs=None,
              predict_difference=False, is_categorical=False, bin_min=None, bin_max=None,
              num_bins=None, quantile_bins=False,
              **kwargs):
    if type(ext_mean) is str: ext_mean = xr.open_dataarray(ext_mean)
    if type(ext_std) is str: ext_std = xr.open_dataarray(ext_std)

    # Open dataset and create data generators
    if cmip:

        # Load vars
        tmp_dict = var_dict.copy()
        constants = tmp_dict.pop('constants')
        ds = xr.merge(
            [xr.open_mfdataset(f'{cmip_dir}/{var}/*.nc', combine='by_coords')
             for var in tmp_dict.keys()] +
            [xr.open_mfdataset(f'{datadir}/constants/*.nc', combine='by_coords')],
            fill_value=0  # For the 'tisr' NaNs
        )
        # pdb.set_trace()
        ds = ds.assign_coords(plev= ds['plev'] / 100)
        ds = ds.rename({'plev': 'level'})
    else:
        ds = xr.merge(
            [xr.open_mfdataset(f'{datadir}/{var}/*.nc', combine='by_coords')
             for var in var_dict.keys()],
            fill_value=0  # For the 'tisr' NaNs
        )

    ds_train = ds.sel(time=slice(*train_years))
    ds_valid = ds.sel(time=slice(*valid_years))
    ds_test = ds.sel(time=slice(*test_years))

    if not only_test:
        dg_train = DataGenerator(
            ds_train, var_dict, lead_time, batch_size=batch_size, output_vars=output_vars,
            data_subsample=data_subsample, norm_subsample=norm_subsample, nt_in=nt_in, dt_in=dt_in,
            mean=ext_mean, std=ext_std, cont_time=cont_time, multi_dt=multi_dt,
            load=train_tfr_files is None,
            tfrecord_files=train_tfr_files,
            tfr_num_parallel_calls=tfr_num_parallel_calls,
            tfr_buffer_size=tfr_buffer_size,
            tfr_prefetch=tfr_prefetch, y_roll=y_roll, X_roll=X_roll, discard_first=discard_first,
            min_lead_time=min_lead_time, tp_log=tp_log, verbose=1, tfr_out=tfr_out,
            tfr_out_idxs=tfr_out_idxs, predict_difference=predict_difference,
            is_categorical=is_categorical, bin_min=bin_min, bin_max=bin_max, num_bins=num_bins, quantile_bins=quantile_bins
        )

        dg_valid = DataGenerator(
            ds_valid, var_dict, lead_time, batch_size=batch_size,
            data_subsample=data_subsample,
            mean=ext_mean if ext_mean is not None else dg_train.mean,
            std=ext_std if ext_std is not None else dg_train.std,
            shuffle=False, output_vars=output_vars, nt_in=nt_in, dt_in=dt_in,
            cont_time=cont_time, multi_dt=multi_dt,
            load=valid_tfr_files is None,
            tfrecord_files=valid_tfr_files,
            tfr_num_parallel_calls=1,
            tfr_buffer_size=1,
            tfr_prefetch=None, y_roll=y_roll, X_roll=X_roll,
            tfr_repeat=False,
            min_lead_time=min_lead_time, tp_log=tp_log, tfr_out=tfr_out,
            tfr_out_idxs=tfr_out_idxs, predict_difference=predict_difference,
            is_categorical=is_categorical, bin_min=bin_min, bin_max=bin_max, num_bins=num_bins, quantile_bins=quantile_bins
        )

    dg_test = DataGenerator(
        ds_test, var_dict, lead_time, batch_size=batch_size,
        data_subsample=data_subsample,
        mean=ext_mean if ext_mean is not None else dg_train.mean,
        std=ext_std if ext_std is not None else dg_train.std,
        shuffle=False, output_vars=output_vars, nt_in=nt_in, dt_in=dt_in,
        cont_time=cont_time, multi_dt=multi_dt,
        load=test_tfr_files is None,
        tfrecord_files=test_tfr_files,
        tfr_num_parallel_calls=1,
        tfr_buffer_size=1,
        tfr_prefetch=None, y_roll=y_roll, X_roll=X_roll,
        tfr_repeat=False,
        min_lead_time=min_lead_time, tp_log=tp_log, tfr_out=tfr_out,
        tfr_out_idxs=tfr_out_idxs, predict_difference=predict_difference,
        is_categorical=is_categorical, bin_min=bin_min, bin_max=bin_max, num_bins=num_bins, 
        quantile_bins=quantile_bins
    )
    if only_test:
        return dg_test
    else:
        if verbose: print(f'Mean = {dg_train.mean}; Std = {dg_train.std}')
        return dg_train, dg_valid, dg_test


def train(datadir, var_dict, output_vars, filters, kernels, lr, batch_size, early_stopping_patience, epochs, exp_id,
         model_save_dir, pred_save_dir, train_years, valid_years, test_years, lead_time, gpu,
         norm_subsample, data_subsample, lr_step, lr_divide, network_type, restore_best_weights,
         bn_position, nt_in, dt_in, use_bias, l2, skip, dropout,
         reduce_lr_patience, reduce_lr_factor, min_lr_times, unres, loss,
         cmip, cmip_dir, pretrained_model, last_pretrained_layer, last_trainable_layer,
         min_es_delta, optimizer, activation, ext_mean, ext_std, cont_time, multi_dt, momentum,
         parametric, one_cycle, long_skip,
         train_tfr_files, valid_tfr_files, test_tfr_files,
         tfr_num_parallel_calls, tfr_buffer_size,
         tfr_prefetch, y_roll, X_roll, discard_first, min_lead_time, relu_idxs, tp_log, tfr_out_idxs,
         predict_difference, is_categorical, bin_min, bin_max, num_bins,
         quantile_bins,
         **kwargs
      ):
    print(type(var_dict))

    # os.environ["CUDA_VISIBLE_DEVICES"]=str(2)
    # # Limit TF memory usage
    # limit_mem()
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join([str(g) for g in gpu])
    mirrored_strategy = tf.distribute.MirroredStrategy(
        devices=[f"/gpu:{i}" for i, g in enumerate(gpu)]
    )

    # Mixed precicion policy
    policy = mixed_precision.Policy('mixed_float16')
    mixed_precision.set_policy(policy)

    # Open dataset and create data generators
    if cmip:
        if len(cmip_dir) > 1:
            dg_train, dg_valid, dg_test = [], [], []
            for cd in cmip_dir:
                dgtr, dgv, dgte = load_data(
                    var_dict, datadir, cmip, cd, train_years, valid_years, test_years,
                    lead_time, batch_size, output_vars, data_subsample, norm_subsample,
                    nt_in, dt_in, ext_mean=ext_mean, ext_std=ext_std, cont_time=cont_time,
                    multi_dt=multi_dt,
                    train_tfr_files=train_tfr_files, valid_tfr_files=valid_tfr_files,
                    test_tfr_files=test_tfr_files,
                    tfr_num_parallel_calls=tfr_num_parallel_calls,
                    tfr_buffer_size=tfr_buffer_size,
                    tfr_prefetch=tfr_prefetch, y_roll=y_roll, X_roll=X_roll, discard_first=discard_first,
                    min_lead_time=min_lead_time, tp_log=tp_log, tfr_out_idxs=tfr_out_idxs,
                    predict_difference=predict_difference,
                    is_categorical=is_categorical, bin_min=bin_min, bin_max=bin_max, num_bins=num_bins,quantile_bins=quantile_bins
                )
                dg_train.append(dgtr); dg_valid.append(dgv); dg_test.append(dgte)
            dg_train, dg_valid, dg_test = [
                CombinedDataGenerator(dg, batch_size) for dg in [dg_train, dg_valid, dg_test]]
        else:
            dg_train, dg_valid, dg_test = load_data(
                var_dict, datadir, cmip, cmip_dir[0], train_years, valid_years, test_years,
                lead_time, batch_size, output_vars, data_subsample, norm_subsample,
                nt_in, dt_in, ext_mean=ext_mean, ext_std=ext_std, cont_time=cont_time,
                multi_dt=multi_dt,
                train_tfr_files=train_tfr_files, valid_tfr_files=valid_tfr_files,
                test_tfr_files=test_tfr_files,
                tfr_num_parallel_calls=tfr_num_parallel_calls,
                tfr_buffer_size=tfr_buffer_size,
                tfr_prefetch=tfr_prefetch, y_roll=y_roll, X_roll=X_roll, discard_first=discard_first,
                min_lead_time=min_lead_time, tp_log=tp_log, tfr_out_idxs=tfr_out_idxs,
                predict_difference=predict_difference,
                is_categorical=is_categorical, bin_min=bin_min, bin_max=bin_max, num_bins=num_bins,quantile_bins=quantile_bins
            )
    else:
        dg_train, dg_valid, dg_test = load_data(
            var_dict, datadir, cmip, cmip_dir, train_years, valid_years, test_years,
            lead_time, batch_size, output_vars, data_subsample, norm_subsample,
            nt_in, dt_in, ext_mean=ext_mean, ext_std=ext_std, cont_time=cont_time,
            multi_dt=multi_dt,
            train_tfr_files=train_tfr_files, valid_tfr_files=valid_tfr_files,
            test_tfr_files=test_tfr_files,
            tfr_num_parallel_calls=tfr_num_parallel_calls,
            tfr_buffer_size=tfr_buffer_size,
            tfr_prefetch=tfr_prefetch, y_roll=y_roll, X_roll=X_roll, discard_first=discard_first,
            min_lead_time=min_lead_time, tp_log=tp_log, tfr_out_idxs=tfr_out_idxs,
            predict_difference=predict_difference,
            is_categorical=is_categorical, bin_min=bin_min, bin_max=bin_max, num_bins=num_bins,quantile_bins=quantile_bins
        )

    # Build model
    if pretrained_model is not None:
        pretrained_model = keras.models.load_model(
            pretrained_model, custom_objects={
                'PeriodicConv2D': PeriodicConv2D,
                'ChannelReLU2D': ChannelReLU2D,
                'lat_mse': keras.losses.mse,
                'lat_mae': keras.losses.mse
            }
        )

    with mirrored_strategy.scope():
        if network_type == 'resnet':
            model = build_resnet(
                filters, kernels, input_shape=dg_train.shape,
                bn_position=bn_position, use_bias=use_bias, l2=l2, skip=skip,
                dropout=dropout, activation=activation, long_skip=long_skip,
                relu_idxs=relu_idxs, categorical=is_categorical, nvars=len(dg_train.output_idxs)
            )
        elif network_type == 'uresnet':
            model = build_uresnet(
                filters, kernels, unres, input_shape=dg_train.shape,
                bn_position=bn_position, use_bias=use_bias, l2=l2, skip=skip,
                dropout=dropout, activation=activation
            )

        if pretrained_model is not None:
            # Copy over weights
            for i, l in enumerate(pretrained_model.layers):
                model.layers[i].set_weights(l.get_weights())
                if l.name == last_pretrained_layer: break

            # Set trainable to false
            if last_trainable_layer is not None:
                for l in model.layers:
                    l.trainable = False
                    if l.name == last_trainable_layer: break
                    
        if multi_dt > 1:
            model = create_multi_dt_model(model, multi_dt, dg_train)

        if loss == 'lat_mse':
            loss = create_lat_mse(dg_train.data.lat)
        if loss == 'lat_mae':
            loss = create_lat_mae(dg_train.data.lat)
        if loss == 'lat_rmse':
            loss = create_lat_rmse(dg_train.data.lat)
        if loss == 'lat_crps':
            loss = create_lat_crps(dg_train.data.lat, len(dg_train.output_idxs))
        if loss == 'lat_crps_relu':
            loss = create_lat_crps(dg_train.data.lat, len(dg_train.output_idxs), relu=True)
        if loss == 'lat_crps_mae':
            loss = create_lat_crps_mae(dg_train.data.lat, len(dg_train.output_idxs))
        if loss == 'lat_crps_lcgev':
            loss = create_lat_crps_lcgev(dg_train.data.lat, len(dg_train.output_idxs))
        if loss == 'lat_log_loss':
            loss = create_lat_log_loss(dg_train.data.lat, len(dg_train.output_idxs))
        if loss == 'lat_categorical_crossentropy':
            loss = create_lat_categorical_loss(dg_train.data.lat, len(dg_train.output_idxs))
            
        if optimizer == 'adam':
            opt = keras.optimizers.Adam(lr)
        elif optimizer =='adadelta':
            opt = keras.optimizers.Adadelta(lr)
        elif optimizer == 'sgd':
            opt = keras.optimizers.SGD(lr, momentum=momentum, nesterov=True)
        elif optimizer == 'rmsprop':
            opt = keras.optimizers.RMSprop(lr, momentum=momentum)

        model.compile(opt, loss)
        print(model.summary())


    # Learning rate settings
    callbacks = []
    if early_stopping_patience is not None:
        callbacks.append(tf.keras.callbacks.EarlyStopping(
          patience=early_stopping_patience,
          verbose=1,
          min_delta=min_es_delta,
          mode='auto',
          restore_best_weights=restore_best_weights
                      ))
    if reduce_lr_patience is not None:
        callbacks.append(tf.keras.callbacks.ReduceLROnPlateau(
            patience=reduce_lr_patience,
            factor=reduce_lr_factor,
            verbose=1,
            min_lr=reduce_lr_factor**min_lr_times*lr,
        ))
    if lr_step is not None:
        callbacks.append(keras.callbacks.LearningRateScheduler(
            LRUpdate(lr, lr_step, lr_divide)
        ))
    if one_cycle:
        callbacks.append(OneCycleLR(
            lr,
            maximum_momentum=None if not optimizer=='sgd' else 0.95,
            minimum_momentum=None if not optimizer=='sgd' else 0.85,
            verbose=1
        ))

    # Train model
    history = model.fit(
        dg_train.tfr_dataset or dg_train,
        epochs=epochs,
        validation_data=dg_valid.tfr_dataset or dg_valid,
        callbacks=callbacks
    )
    print(f'Saving model: {model_save_dir}/{exp_id}.h5')
    model.save(f'{model_save_dir}/{exp_id}.h5')
    print(f'Saving model weights: {model_save_dir}/{exp_id}_weights.h5')
    model.save_weights(f'{model_save_dir}/{exp_id}_weights.h5')
    print(f'Saving training_history: {model_save_dir}/{exp_id}_history.pkl')
    to_pickle(history.history, f'{model_save_dir}/{exp_id}_history.pkl')
    print(f'Saving norm files: {model_save_dir}/{exp_id}_mean.nc and {model_save_dir}/{exp_id}_std.nc')
    dg_train.mean.to_netcdf(f'{model_save_dir}/{exp_id}_mean.nc')
    dg_train.std.to_netcdf(f'{model_save_dir}/{exp_id}_std.nc')

    # Create predictions
    preds = create_predictions(model, dg_test, parametric=parametric, multi_dt=multi_dt>1)
    if len(preds.lat) != 32:
        preds = regrid(preds, ddeg_out=5.625)
    print(f'Saving predictions: {pred_save_dir}/{exp_id}.nc')
    preds.to_netcdf(f'{pred_save_dir}/{exp_id}.nc')

    # Print score in real units

    if not cmip:
        if '5.625deg' in datadir:
            valdir = datadir
        else:
            valdir = '/'.join(datadir.split('/')[:-2] + ['5.625deg/'])

        z500_valid = load_test_data(f'{valdir}geopotential_500', 'z', years=slice(test_years[0], test_years[1])).drop('level')
        t850_valid = load_test_data(f'{valdir}temperature_850', 't', years=slice(test_years[0], test_years[1])).drop('level')
        tp = xr.open_mfdataset(f'{valdir}/6hr_precipitation/*.nc', combine='by_coords').sel(
            time=slice(test_years[0], test_years[1]))
        t2m = xr.open_mfdataset(f'{valdir}/2m_temperature/*.nc', combine='by_coords').sel(
            time=slice(test_years[0], test_years[1]))
        valid = xr.merge([z500_valid, t850_valid, tp, t2m])

        print(compute_weighted_rmse(
            preds, valid
        ).load())



def load_args(my_config=None):
    p = ArgParser()
    p.add_argument('-c', '--my-config', is_config_file=True, help='config file path', default=my_config)
    p.add_argument('--datadir', type=str, required=True, help='Path to data')
    p.add_argument('--exp_id', type=str, required=True, help='Experiment identifier')
    p.add_argument('--model_save_dir', type=str, required=True, help='Path to save model')
    p.add_argument('--pred_save_dir', type=str, required=True, help='Path to save predictions')
    p.add_argument('--var_dict', required=True, help='Variables: as an ugly dictionary...')
    p.add_argument('--output_vars', nargs='+', help='Output variables. Format {var}_{level}', default=None)
    p.add_argument('--filters', type=int, nargs='+', required=True, help='Filters for each layer')
    p.add_argument('--kernels', type=int, nargs='+', default=None, help='Kernel size for each layer')
    p.add_argument('--lead_time', type=int, required=True, help='Forecast lead time')
    # p.add_argument('--iterative', type=bool, default=False, help='Is iterative forecast')
    # p.add_argument('--iterative_max_lead_time', type=int, default=5*24, help='Max lead time for iterative forecasts')
    p.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    p.add_argument('--activation', type=str, default='relu', help='Activation function')
    p.add_argument('--batch_size', type=int, default=64, help='batch_size')
    p.add_argument('--epochs', type=int, default=1000, help='epochs')
    p.add_argument('--optimizer', type=str, default='adam', help='Optimizer')
    p.add_argument('--early_stopping_patience', type=int, default=None, help='Early stopping patience')
    p.add_argument('--min_es_delta', type=float, default=0, help='Minimum improvement for early stopping')
    p.add_argument('--restore_best_weights', type=int, default=1, help='ES parameter')
    p.add_argument('--reduce_lr_patience', type=int, default=None, help='Reduce LR patience')
    p.add_argument('--reduce_lr_factor', type=float, default=0.2, help='Reduce LR factor')
    p.add_argument('--min_lr_times', type=int, default=1, help='Reduce LR patience')
    p.add_argument('--lr_step', type=int, default=None, help='LR decay step')
    p.add_argument('--lr_divide', type=int, default=None, help='LR decay division factor')
    p.add_argument('--loss', type=str, default='mse', help='Loss function')
    p.add_argument('--train_years', type=str, nargs='+', default=('1979', '2015'), help='Start/stop years for training')
    p.add_argument('--valid_years', type=str, nargs='+', default=('2016', '2016'),
                   help='Start/stop years for validation')
    p.add_argument('--test_years', type=str, nargs='+', default=('2017', '2018'), help='Start/stop years for testing')
    p.add_argument('--data_subsample', type=int, default=1, help='Subsampling for training data')
    p.add_argument('--norm_subsample', type=int, default=1, help='Subsampling for mean/std')
    p.add_argument('--gpu', type=int, default=0, help='Which GPU', nargs='+')
    p.add_argument('--network_type', type=str, default='resnet', help='Type')
    p.add_argument('--bn_position', type=str, default=None, help='pre, mid or post')
    p.add_argument('--nt_in', type=int, default=1, help='Number of input time steps')
    p.add_argument('--dt_in', type=int, default=1, help='Time step of intput time steps (after subsampling)')
    p.add_argument('--use_bias', type=int, default=1, help='Use bias in resnet convs')
    p.add_argument('--l2', type=float, default=0, help='Weight decay')
    p.add_argument('--momentum', type=float, default=0.9, help='Momentum')
    p.add_argument('--dropout', type=float, default=0, help='Dropout')
    p.add_argument('--skip', type=int, default=1, help='Add skip convs in resnet builder')
    # p.add_argument('--u_skip', type=int, default=1, help='Add skip convs in unet')
    # p.add_argument('--unet_layers', type=int, default=5, help='Number of unet layers')
    p.add_argument('--unres', type=int, default=2, nargs='+', help='Resblocks per unet partition')
    p.add_argument('--cmip', type=int, default=0, help='Is CMIP')
    p.add_argument('--cmip_dir', type=str, default=None, nargs='+', help='Dirs for CMIP data')
    p.add_argument('--pretrained_model', type=str, default=None, help='Path to pretrained model')
    p.add_argument('--last_pretrained_layer', type=str, default=None, help='Name of last pretrained layer')
    p.add_argument('--last_trainable_layer', type=str, default=None, help='Name of last trainable layer')
    p.add_argument('--ext_mean', type=str, default=None, help='External normalization mean')
    p.add_argument('--ext_std', type=str, default=None, help='External normalization std')
    p.add_argument('--cont_time', type=int, default=0, help='Continuous time 0/1')
    p.add_argument('--multi_dt', type=int, default=1, help='Differentiate through multiple time steps')
    p.add_argument('--parametric', type=int, default=0, help='Is parametric')
    p.add_argument('--one_cycle', type=int, default=0, help='Use OneCycle LR')
    # p.add_argument('--las_kernel', type=int, default=None, help='')
    # p.add_argument('--las_gauss_std', type=int, default=None, help='')
    p.add_argument('--long_skip', type=int, default=0, help='Use long skip connections 0/1')

    p.add_argument('--train_tfr_files', type=str, default=None, help='TFR regex')
    p.add_argument('--valid_tfr_files', type=str, default=None, help='TFR regex')
    p.add_argument('--test_tfr_files', type=str, default=None, help='TFR regex')
    p.add_argument('--tfr_num_parallel_calls', type=int, default=1, help='')
    p.add_argument('--tfr_buffer_size', type=int, default=100, help='')
    p.add_argument('--tfr_prefetch', type=int, default=None, help='')

    p.add_argument('--min_lead_time', type=int, default=None, help='for cont.')
    p.add_argument('--y_roll', type=int, default=None, help='In hours.')
    p.add_argument('--X_roll', type=int, default=None, help='In hours.')
    p.add_argument('--discard_first', type=int, default=None, help='Discard first x time steps in train generator')

    p.add_argument('--relu_idxs', type=int, default=None, nargs='+', help='for tp')
    p.add_argument('--tp_log', type=float, default=None, help='for tp')

    p.add_argument('--tfr_out', type=int, default=0, help='output all times up to lead time')
    p.add_argument('--tfr_out_idxs', type=int, default=None, nargs='+', help='out idxs')

    p.add_argument('--predict_difference', type=int, default=0, help='')
    p.add_argument('--is_categorical', type=int, default=0, help='')
    p.add_argument('--bin_min', type=float, default=None, help='')
    p.add_argument('--bin_max', type=float, default=None, help='')
    p.add_argument('--num_bins', type=int, default=None, help='')
    p.add_argument('--quantile_bins', type=int, default=0, help='')

    args = p.parse_args() if my_config is None else p.parse_args(args=[])
    args.var_dict = ast.literal_eval(args.var_dict)
    return vars(args)