import fire
from fire import Fire
import xarray as xr
import numpy as np
from src.data_generator import *
from src.train import *
from src.networks import *
from src.utils import *
from tensorflow.keras import backend as K
#ToDO: 
#load full data instead of batches. output for full size of X. Stii unable to do. How to I run a code on GPU?
#make it work for all networks. #(Differences: custom_objects, loss function, ...
#CouldDo: pass optional arguments. like is_normalized, start_date, end_date


def get_input(args, exp_id_path, datadir, model_save_dir):
    #essential arguments
    args=load_args(exp_id_path)
    exp_id=args['exp_id']
    var_dict=args['var_dict']
    batch_size=args['batch_size']
    output_vars=args['output_vars']
    
    #optional inputs. see load_args() for default values.
    data_subsample=args['data_subsample']
    norm_subsample=args['norm_subsample']
    nt_in=args['nt_in'] #Ques: sometimes mentioned in config file as 'nt'. is that same?
    #nt_in=args['nt']
    dt_in=args['dt_in']
    test_years=args['test_years']
    lead_time=args['lead_time']
    
    #changing paths
#   model_save_dir='/home/garg/data/WeatherBench/predictions/saved_models'
#   datadir='/home/garg/data/WeatherBench/5.625deg'
    
    #run data generator
    ds = xr.merge([xr.open_mfdataset(f'{datadir}/{var}/*.nc', combine='by_coords') 
                   for var in var_dict.keys()])
    mean = xr.open_dataarray(f'{model_save_dir}/{exp_id}_mean.nc') 
    std = xr.open_dataarray(f'{model_save_dir}/{exp_id}_std.nc')
    
    ds_test= ds.sel(time=slice(test_years[0],test_years[-1]))
    #Ques: Should shuffle be False? since its testing.
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
    model=tf.keras.models.load_model(saved_model_path,
                                     custom_objects={'PeriodicConv2D':PeriodicConv2D,
                                    'lat_mse': tf.keras.losses.mse})
    #ToDo: add other loss functions to custom_objects. doesn't matter if it is not used in the model itself, only so that load_model() doesn't break)
    #Since we dont build again, we dont need to pass model params like kernel, filters, activation, dropout,loss and other details to the network.

    #model.summary()
    return model

def predict(dg_test,model,number_of_forecasts, output_vars):
    #For just 1 batch of data
#     X,y=dg_test[0] #currently limiting output due to RAM issues.  
#     #test-time dropout
#     func = K.function(model.inputs + [K.learning_phase()], model.outputs)
#     pred_ensemble = np.array([np.asarray(func([X] + [1.]), dtype=np.float32).squeeze() for _ in
#                               range(number_of_forecasts)])
    
    #For Full Data. @Stephan: Please check.
    #Issue: The last batch is shorter (18 elements instead of 32). so list has differing sizes. 
    #so unable to convert to np.array(preds). error: can't broadcast from shape (1,32,32,64,2)   to (2)
    #so using an if conditon to break.
    
    #NOTE: Always better to append in a list, rather than as numpy array. bcoz numpy array equires contiguous memory (not exactly, but yeah), so if you keep appending it would be slow. alternatively, pre-allocate an empty numpy array. Current method: append to list--> convert to numpy array-->reshape.
    func = K.function(model.inputs + [K.learning_phase()], model.outputs)
    preds = []
    counter=0
    for X, y in dg_test: 
        preds.append(np.array([np.asarray(func([X] + [1.]), dtype=np.float32).squeeze() 
                               for _ in range(number_of_forecasts)]))

        if (counter%10==0):
                print(counter)
        if counter==len(dg_test)-2:
            print(counter)
            break
        counter=counter+1

    pred_ensemble=np.array(preds)
    #reshaping. Be careful!
    shp=pred_ensemble.shape
    pred_ensemble=pred_ensemble.transpose(1,0,2,3,4,5).reshape(shp[1],-1,shp[-3],shp[-2],shp[-1])
    pred_ensemble.shape

    #for last batch (Bad method)
    last_element=len(dg_test)-1
    X,y=dg_test[last_element]
    pred_last=np.array([np.asarray(func([X] + [1.]), dtype=np.float32).squeeze() 
                                for _ in range(number_of_forecasts)])
    pred_ensemble=np.append(pred_ensemble,pred_last,axis=1)
    
    #unnormalize
    #observation=y
#     observation=(observation * dg_test.std.isel(level=dg_test.output_idxs).values +
#          dg_test.mean.isel(level=dg_test.output_idxs).values)
    pred_ensemble=(pred_ensemble * dg_test.std.isel(level=dg_test.output_idxs).values +
         dg_test.mean.isel(level=dg_test.output_idxs).values)
    
    #numpy --> xarray.
    preds = xr.Dataset()
    i=0
    for var in output_vars:
        da= xr.DataArray(pred_ensemble[...,i], coords={'member': np.arange(number_of_forecasts),'time': dg_test.data.time.isel(time=slice(None,pred_ensemble.shape[1])), 'lat': dg_test.data.lat, 'lon': dg_test.data.lon,}, dims=['member', 'time','lat', 'lon'])
        
        preds[var]=da
        i=i+1
        
    return preds

def main(number_of_forecasts, exp_id_path, datadir, model_save_dir, pred_save_dir):
    args=load_args(exp_id_path)
    exp_id=args['exp_id']
    output_vars=args['output_vars']
    
    os.environ["CUDA_VISIBLE_DEVICES"]=str(0)
    limit_mem()
    dg_test=get_input(args, exp_id_path, datadir, model_save_dir)
    mymodel=get_model(exp_id, model_save_dir)
    preds=predict(dg_test,mymodel,number_of_forecasts, output_vars)
    
    
    #changing paths
#     model_save_dir='/home/garg/data/WeatherBench/predictions/saved_models'
#     datadir='/home/garg/data/WeatherBench/5.625deg'
#     pred_save_dir='/home/garg/data/WeatherBench/predictions'
    
    preds.to_netcdf(f'{pred_save_dir}/{exp_id}.nc')
    print(f'saved on disk in {pred_save_dir}/{exp_id}.nc')
    return

    

if __name__ == '__main__':
      fire.Fire(main)