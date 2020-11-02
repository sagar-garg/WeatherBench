import fire
from fire import Fire
import xarray as xr
import numpy as np
from src.data_generator import *
from src.train import *
from src.networks import *
from src.utils import *
#from tensorflow.keras import backend as K #slow method
import tqdm
from tqdm import tqdm
#ToDO: 
#make it work for all networks. #(Differences: custom_objects, loss function, ...
#remove extraneous info displayed.
#add a descripton of the function
#pass optional arguments: start_date, end_date


def get_input(args):
    #essential arguments
    args['ext_mean'] =xr.open_dataarray(f'{args["model_save_dir"]}/{args["exp_id"]}_mean.nc')
    args['ext_std'] = xr.open_dataarray(f'{args["model_save_dir"]}/{args["exp_id"]}_std.nc')
    dg_test=load_data(**args, only_test=True)
    return dg_test

def get_model(args):
    
    tf.compat.v1.disable_eager_execution() #needed
    
    model = keras.models.load_model(
    f'{args["model_save_dir"]}/{args["exp_id"]}.h5',
    custom_objects={'PeriodicConv2D': PeriodicConv2D, 'ChannelReLU2D': ChannelReLU2D, 
                   'lat_mse': tf.keras.losses.mse})
#RECHECK LOSS fn.!
    
    #adding dropout
    c = model.get_config()
    for l in c['layers']:
        if l['class_name'] == 'Dropout':
            l['inbound_nodes'][0][0][-1] = {'training': True}
    
    model2 = keras.models.Model.from_config(c, custom_objects={'PeriodicConv2D': PeriodicConv2D, 'ChannelReLU2D': ChannelReLU2D, 'lat_mse': tf.keras.losses.mse})
    model2.set_weights(model.get_weights())
            
    #ToDo: add other loss functions to custom_objects. doesn't matter if it is not used in the model itself, only so that load_model() doesn't break)
    
    return model2

def predict(dg, model,ensemble_size, multi_dt=False, verbose=0, no_mean=False):
#     os.environ["CUDA_VISIBLE_DEVICES"]=str(0)
#     policy = mixed_precision.Policy('mixed_float16')
#     mixed_precision.set_policy(policy)
    
    level_names = dg.data.isel(level=dg.output_idxs).level_names
    level = dg.data.isel(level=dg.output_idxs).level
    
    preds = []
    for _ in tqdm(range(ensemble_size)):
        preds.append(model.predict(dg.tfr_dataset or dg, verbose=verbose))
    
    preds = np.array(preds)
    

    preds = xr.DataArray(
        preds[0] if multi_dt else preds,
        dims=['member','time', 'lat', 'lon', 'level'],
        coords={'member':np.arange(ensemble_size),'time': dg.valid_time, 'lat': dg.data.lat, 'lon': dg.data.lon,
                'level': level,
                'level_names': level_names
                },
    )
    # Unnormalize
    mean = dg.mean.isel(level=dg.output_idxs).values if not no_mean else 0
    std = dg.std.isel(level=dg.output_idxs).values
    preds = preds * std + mean
    
    unique_vars = list(set([l.split('_')[0] for l in preds.level_names.values]))

    # Reverse tranforms
    if hasattr(dg.mean, 'tp_log') and 'tp' in unique_vars:
        tp_idx = list(preds.level_names).index('tp')
        preds.values[..., tp_idx] = log_retrans(preds.values[..., tp_idx], dg.mean.tp_log)

    das = []
    for v in unique_vars:
        idxs = [i for i, vv in enumerate(preds.level_names.values) if vv.split('_')[0] == v]
        da = preds.isel(level=idxs).squeeze().drop('level_names')
        if not 'level' in da.dims: da = da.drop('level')
        das.append({v: da})
    return(xr.merge(das))    

def main(ensemble_size, exp_id_path, datadir, model_save_dir, pred_save_dir, data_subsample=1, dt=6, gpu=0):
    
    os.environ["CUDA_VISIBLE_DEVICES"]=str(gpu)
    policy = mixed_precision.Policy('mixed_float16')
    mixed_precision.set_policy(policy)
    
    
    args=load_args(exp_id_path)
    args['model_save_dir']=model_save_dir
    args['pred_save_dir']=pred_save_dir
    args['datadir']=datadir
    args['data_subsample'] = data_subsample
    args['dt'] = dt
    args['train_tfr_files'] = None
    args['valid_tfr_files'] = None
    args['test_tfr_files'] = None
    #args['test_years']=['2018-12-01','2018-12-31']
    dg_test=get_input(args)
    mymodel=get_model(args)
    preds=predict(dg_test,mymodel,ensemble_size, multi_dt=False, verbose=0, no_mean=False)
    
    
    #changing paths
#     model_save_dir='/home/garg/data/WeatherBench/predictions/saved_models'
#     datadir='/home/garg/data/WeatherBench/5.625deg'
#     pred_save_dir='/home/garg/data/WeatherBench/predictions'
    
    preds.to_netcdf(f'{args["pred_save_dir"]}/{args["exp_id"]}_m{ensemble_size}.nc')
    print(f'saved on disk in {args["pred_save_dir"]}/{args["exp_id"]}_m{ensemble_size}.nc')
    return

    

if __name__ == '__main__':
      fire.Fire(main)