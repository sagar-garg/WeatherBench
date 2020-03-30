"""
Functions for evaluating forecasts.
"""
import numpy as np
import xarray as xr

def load_test_data(path, var, years=slice('2017', '2018'), cmip=False):
    """
    Args:
        path: Path to nc files
        var: variable. Geopotential = 'z', Temperature = 't'
        years: slice for time window

    Returns:
        dataset: Concatenated dataset for 2017 and 2018
    """
    assert var in ['z', 't'], 'Test data only for Z500 and T850'
    ds = xr.open_mfdataset(f'{path}/*.nc', combine='by_coords')[var]
    if cmip:
        ds['plev'] /= 100
        ds = ds.rename({'plev': 'level'})
    try:
        ds = ds.sel(level=500 if var == 'z' else 850).drop('level')
    except ValueError:
        pass
    return ds.sel(time=years)

def compute_weighted_rmse(da_fc, da_true, mean_dims=xr.ALL_DIMS):
    """
    Compute the RMSE with latitude weighting from two xr.DataArrays.

    Args:
        da_fc (xr.DataArray): Forecast. Time coordinate must be validation time.
        da_true (xr.DataArray): Truth.

    Returns:
        rmse: Latitude weighted root mean squared error
    """
    error = da_fc - da_true
    weights_lat = np.cos(np.deg2rad(error.lat))
    weights_lat /= weights_lat.mean()
    rmse = np.sqrt(((error)**2 * weights_lat).mean(mean_dims))
    if type(rmse) is xr.Dataset:
        rmse = rmse.rename({v: v + '_rmse' for v in rmse})
    else: # DataArray
        rmse.name = error.name + '_rmse' if not error.name is None else 'rmse'
    return rmse

def evaluate_iterative_forecast(fc_iter, da_valid):
    rmses = []
    for lead_time in fc_iter.lead_time:
        fc = fc_iter.sel(lead_time=lead_time)
        fc['time'] = fc.time + np.timedelta64(int(lead_time), 'h')
        rmses.append(compute_weighted_rmse(fc, da_valid))
    return xr.concat(rmses, 'lead_time')
    # return xr.DataArray(rmses, dims=['lead_time'], coords={'lead_time': fc_iter.lead_time})


def compute_weighted_acc(da_fc, da_true):
    clim = da_true.mean('time')
    t = np.intersect1d(da_fc.time, da_true.time)
    fa = da_fc.sel(time=t) - clim
    a = da_true.sel(time=t) - clim
    
    weights_lat = np.cos(np.deg2rad(da_fc.lat))
    weights_lat /= weights_lat.mean()
    w = weights_lat
    
    fa_prime = fa - fa.mean()
    a_prime = a - a.mean()
    
    acc = (
        np.sum(w * fa_prime * a_prime) /
        np.sqrt(
            np.sum(w * fa_prime**2) * np.sum(w * a_prime**2)
        )
    )
    return acc

def compute_weighted_meanspread(prediction):
    """
    input: xarray. Coordinates: input_number, ensemble_number, lat, lon. Variables: z500, t850 or anything else
    input_number: Let there be I initial conditions
    ensemble_number: For each initial condition, let there be N forecasts (Toimprove: N can be differnet for differnet input_number)

    #mean variance
    #1. for each input i, for each gridpoint, find variance among all N forecasts for that single input i
    #2. for each input i, find latitude-weighted average of all the lat*lon points
    #3. find average of all I inputs. take square root
    """
    #ToDO: add assert condition to check for input size. Alternatively, if input does not have 'input_number' then add it as dimension
    #ToDo: change from names like 'forecast_number' to dim=0 or dim=1 in order to generalize (Beneficial??)
    var1=prediction.std('forecast_number')
    weights_lat = np.cos(np.deg2rad(var1.lat))
    weights_lat /= weights_lat.mean()
    var2 =  (var1*weights_lat).mean(dim={'lat','lon'})
    #var2=var1.mean(dim={'lat','lon'}) #without weighted latitude.
    var3=var2.mean('input_number') #change to call it time/ valid_time?
    mean_spread=np.sqrt(var3)
    
    if type(mean_spread) is xr.Dataset:
        mean_spread = mean_spread.rename({v: v + '_mean_spread' for v in mean_spread})
    else: # DataArray
        mean_spread.name = error.name + '_mean_spread' if not error.name is None else 'mean_spread'
    return mean_spread

def compute_weighted_meanrmse(observation,prediction):
    """
    similar to spread. 
    1. for each input i, find mean of N forecasts:- ensmean
    2. error=(ensmean-observation)**2
    3. for each input i, find average error over grid.
    4. find average among all I inputs
    5. take square root
    """
    #computing mean rmse over all forecasts for all inputs
    
    ensmean1=prediction.mean('forecast_number')
    error1=(ensmean1-observation)**2
    weights_lat = np.cos(np.deg2rad(error1.lat))
    weights_lat /= weights_lat.mean()
    error2 = (error1* weights_lat).mean(dim={'lat','lon'})
    #error2=error1.mean(dim={'lat','lon'}) #without weighted lat.
    mean_rmse=np.sqrt(error2.mean('input_number'))
    if type(mean_rmse) is xr.Dataset:
            mean_rmse = mean_rmse.rename({v: v + '_mean_rmse' for v in mean_rmse})
    else: # DataArray
        mean_rmse.name = mean_rmse.name + '_mean_rmse' if not error.name is None else 'mean_rmse'
    return mean_rmse

def crps_score(observation,prediction,forecast_axis): 
    #ToDo: improve argument that tells which dim is forecast_number
    #ToDo: add weightd averaging in the code. so as to ouput just 1 value instead of a dataset.
    import properscoring as ps
    obs = np.asarray(observation.to_array(), dtype=np.float32).squeeze();
    #shape: (variable,input_number, lat, lon)
    pred=np.asarray(prediction.to_array(), dtype=np.float32).squeeze();
    #shape: (variable, forecast_number, input_number, lat, lon)
    crps=ps.crps_ensemble(obs,pred, weights=None, issorted=False,axis=forecast_axis) 
    #axis of forecast_number. crps.shape  #(variable, input_number, lat, lon)
    
    #Converting back to xarray
    crps_score = xr.Dataset({
        'z500': xr.DataArray(
        crps[0,...],
        dims=['input_number', 'lat', 'lon'],
        coords={'input_number': observation.input_number, 'lat': observation.lat, 'lon': observation.lon,
                },
        ),
        't850': xr.DataArray(
        crps[1,...],
        dims=['input_number', 'lat', 'lon'],
        coords={'input_number': observation.input_number, 'lat': observation.lat, 'lon': observation.lon,
                },
        )
    })
    return crps_score 