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


def get_input(args, exp_id_path, datadir, model_save_dir,start_date=None,end_date=None):
    #essential arguments
    args=load_args(exp_id_path)
    exp_id=args['exp_id']
    var_dict=args['var_dict']
    batch_size=args['batch_size']
    output_vars=args['output_vars']
    
    #optional inputs. see load_args() for default values.
    data_subsample=args['data_subsample']
    norm_subsample=args['norm_subsample'] #doesnt matter since we pass external mean/std
    nt_in=args['nt_in']
    #nt_in=args['nt']
    dt_in=args['dt_in']
    lead_time=args['lead_time']
    test_years=args['test_years']
    
    #changing paths
#   model_save_dir='/home/garg/data/WeatherBench/predictions/saved_models'
#   datadir='/home/garg/data/WeatherBench/5.625deg'
    
    #run data generator
    ds = xr.merge([xr.open_mfdataset(f'{datadir}/{var}/*.nc', combine='by_coords') 
                   for var in var_dict.keys()])
    mean = xr.open_dataarray(f'{model_save_dir}/{exp_id}_mean.nc') 
    std = xr.open_dataarray(f'{model_save_dir}/{exp_id}_std.nc')
    
    if (start_date and end_date)!=None:
        ds_test=ds.sel(time=slice(start_date,end_date))
    else:
        ds_test= ds.sel(time=slice(test_years[0],test_years[-1]))
    #Shuffle should be false. nt_in, data_subsample needed and should not be changed.
    dg_test = DataGenerator(ds_test, var_dict, lead_time, batch_size=batch_size, shuffle=False,
                            load=True,mean=mean, std=std, output_vars=output_vars,
                            data_subsample=data_subsample, norm_subsample=norm_subsample,
                            nt_in=nt_in,dt_in=dt_in) 
    return dg_test

def get_model(exp_id, model_save_dir):
    tf.compat.v1.disable_eager_execution() #needed
    saved_model_path=f'{model_save_dir}/{exp_id}.h5'
    
    substr=['resnet','unet_google','unet']
    assert any(x in exp_id for x in substr)
    model=tf.keras.models.load_model(saved_model_path,custom_objects={'PeriodicConv2D':PeriodicConv2D,
                                    'lat_mse': tf.keras.losses.mse})
    #adding dropout
    c = model.get_config()
    for l in c['layers']:
        if l['class_name'] == 'Dropout':
            l['inbound_nodes'][0][0][-1] = {'training': True}
    
    model2 = keras.models.Model.from_config(c, custom_objects={'PeriodicConv2D':PeriodicConv2D,'lat_mse':tf.keras.losses.mse})
    model2.set_weights(model.get_weights())
            
    #ToDo: add other loss functions to custom_objects. doesn't matter if it is not used in the model itself, only so that load_model() doesn't break)
    #model2.summary()
    return model2

def predict(dg_test,model,ensemble_size, output_vars):
    
    preds = []
    for _ in tqdm(range(ensemble_size)):
        preds.append(model.predict(dg_test))
    
    pred_ensemble = np.array(preds)
    #unnormalize
    pred_ensemble=(pred_ensemble * dg_test.std.isel(level=dg_test.output_idxs).values +
                   dg_test.mean.isel(level=dg_test.output_idxs).values)
    
    #numpy --> xarray.
    preds = xr.Dataset()
    for i,var in enumerate(output_vars):
        da= xr.DataArray(pred_ensemble[...,i], 
                         coords={'member': np.arange(ensemble_size),
                                 'time': dg_test.data.time.sel(time=dg_test.valid_time),
                                 'lat': dg_test.data.lat, 'lon': dg_test.data.lon,}, 
                         dims=['member', 'time','lat', 'lon'])
        preds[var]=da
        
    return preds

def main(ensemble_size, exp_id_path, datadir, model_save_dir, pred_save_dir, start_date=None,end_date=None):
    args=load_args(exp_id_path)
    exp_id=args['exp_id']
    output_vars=args['output_vars']
    
    dg_test=get_input(args, exp_id_path, datadir, model_save_dir, start_date,end_date)
    mymodel=get_model(exp_id, model_save_dir)
    preds=predict(dg_test,mymodel,ensemble_size, output_vars)
    
    
    #changing paths
#     model_save_dir='/home/garg/data/WeatherBench/predictions/saved_models'
#     datadir='/home/garg/data/WeatherBench/5.625deg'
#     pred_save_dir='/home/garg/data/WeatherBench/predictions'
    
    preds.to_netcdf(f'{pred_save_dir}/{exp_id}.nc')
    print(f'saved on disk in {pred_save_dir}/{exp_id}.nc')
    return

    

if __name__ == '__main__':
      fire.Fire(main)