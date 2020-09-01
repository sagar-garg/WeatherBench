import re
import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
import xarray as xr
import datetime
import pandas as pd
import pdb
from tqdm import tqdm
from glob import glob

datadir = '/data/stephan/WeatherBench/5.625deg/'
var_dict = {
    'geopotential': ('z', [50, 250, 500, 600, 700, 850, 925]), 
    'temperature': ('t', [50, 250, 500, 600, 700, 850, 925]), 
    'u_component_of_wind': ('u', [50, 250, 500, 600, 700, 850, 925]), 
    'v_component_of_wind': ('v', [50, 250, 500, 600, 700, 850, 925]), 
    'specific_humidity': ('q', [50, 250, 500, 600, 700, 850, 925]), 
    'toa_incident_solar_radiation': ('tisr', None), 
    '2m_temperature': ('t2m', None), 
    '6hr_precipitation': ('tp', None), 
    'constants': ['lsm','orography','lat2d']
}
output_vars = ['z_500', 't_850', 't2m']
lead_time = 72
data_subsample = 2
norm_subsample = 30000
nt = 3
dt = 6
discard_first = 24

ds = xr.merge(
    [xr.open_mfdataset(f'{datadir}/{var}/*.nc', combine='by_coords')
     for var in var_dict.keys()],
    fill_value=0  # For the 'tisr' NaNs
)

train_years = ['1979', '2015']  # For full training data, use ['1979', '2015']. Will use 200GB of RAM.
valid_years = ['2016', '2016']
test_years = ['2017', '2018']
ds_train = ds.sel(time=slice(*train_years))
ds_valid = ds.sel(time=slice(*valid_years))
ds_test = ds.sel(time=slice(*test_years))

dg_train = DataGenerator(
    ds_train,
    var_dict,
    lead_time,
    output_vars=output_vars,
    data_subsample=data_subsample,
    norm_subsample=norm_subsample,
    nt_in=nt,
    dt_in=dt,
    discard_first=discard_first
)
dg_valid = DataGenerator(
    ds_valid,
    var_dict,
    lead_time,
    output_vars=output_vars,
    data_subsample=data_subsample,
    norm_subsample=norm_subsample,
    nt_in=nt,
    dt_in=dt,
    discard_first=discard_first,
    mean=dg_train.mean,  # Remember to use same mean and std for normalization
    std=dg_train.std,
    shuffle=False
)

dg_test = DataGenerator(
    ds_test,
    var_dict,
    lead_time,
    output_vars=output_vars,
    data_subsample=data_subsample,
    norm_subsample=norm_subsample,
    nt_in=nt,
    dt_in=dt,
    discard_first=discard_first,
    mean=dg_train.mean,  # Remember to use same mean and std for normalization
    std=dg_train.std,
    shuffle=False
)

numpy_dir = '/home/youxie/Weather/test_0025/test/'

files = sorted(glob(f'{numpy_dir}predict_test*.npy'))

arrs = []
for f in files:
    arrs.append(np.load(f))
preds = np.array(arrs)

dg = dg_test

level_names = dg.data.isel(level=dg.output_idxs).level_names
level = dg.data.isel(level=dg.output_idxs).level

mean = dg.mean.isel(level=dg.output_idxs).values
std = dg.std.isel(level=dg.output_idxs).values

valid_time = dg.valid_time[:len(preds)]

preds = xr.DataArray(
    preds,
    dims=['time', 'lat', 'lon', 'level'],
    coords={'time': valid_time, 'lat': dg.data.lat, 'lon': dg.data.lon,
            'level': level,
            'level_names': level_names
            },
)

preds = preds * std + mean

unique_vars = list(set([l.split('_')[0] for l in preds.level_names.values]))

das = []
for v in unique_vars:
    idxs = [i for i, vv in enumerate(preds.level_names.values) if vv.split('_')[0] == v]
    da = preds.isel(level=idxs).squeeze().drop('level_names')
    if not 'level' in da.dims: da = da.drop('level')
    das.append({v: da})
preds =  xr.merge(das)

print(preds)

z500_valid = xr.open_mfdataset(f'{datadir}geopotential_500/*.nc').sel(time=slice('2016', '2018', None)).drop('level')
t850_valid = xr.open_mfdataset(f'{datadir}temperature_850/*.nc').sel(time=slice('2016', '2018', None)).drop('level')
t2m_valid = xr.open_mfdataset(f'{datadir}2m_temperature/*.nc').sel(time=slice('2016', '2018', None))
tp_valid = xr.open_mfdataset(f'{datadir}6hr_precipitation/*.nc').sel(time=slice('2016', '2018', None))

valid = xr.merge([z500_valid, t850_valid, t2m_valid, tp_valid]).load()

print(valid)

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
    return rmse

rmse = compute_weighted_rmse(preds, valid)
print(rmse)