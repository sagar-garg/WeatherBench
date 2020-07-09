from .utils import *
import xarray as xr
from fire import Fire
"""
2m_temperature_daily
python -m src.compute_avg_agg /data/stephan/WeatherBench/CMIP/MPI-ESM_r2/5.625deg/2m_temperature/ /data/stephan/WeatherBench/CMIP/MPI-ESM_r2/5.625deg/2m_temperature_daily/2m_temperature_daily_ _5.625deg.nc 24 --save_years 5
"""

def compute_avg_agg(datadir, savepref, savesuf, a_hours=24, method='avg', save_years=1, rename=None, center=False):
    ds_in = xr.open_mfdataset(f'{datadir}/*.nc', combine='by_coords')
    dt = ds_in.time.diff('time')[0].values / np.timedelta64(1, 'h')
    nt = int(a_hours / dt)
    if method == 'avg':
        ds_out = ds_in.rolling(time=nt, center=center, keep_attrs=True).mean()
    elif method == 'agg':
        ds_out = ds_in.rolling(time=nt, center=center, keep_attrs=True).sum()
    elif method == 'pr_cmip':
        ds_out = ds_in.assign_coords({'time': ds_in['time'] + np.timedelta64(int(dt / 2), 'h')})
        ds_out.pr.values = ds_out.pr / 997 * 60 * 60 * 6
    if rename is not None:
        ds_out = ds_out.rename(rename)
    year_range = (ds_out.time.dt.year.min().values, ds_out.time.dt.year.max().values + 1)
    for y in range(*year_range, save_years):
        savefn = f'{savepref}{y if save_years == 1 else str(y) + "_" + str(min(y + save_years - 1, year_range[-1]))}{savesuf}'
        ds_save = ds_out.sel(time=slice(str(y), str(y + save_years - 1))).load()
        print(f'Saving {savefn}')
        ds_save.to_netcdf(savefn)

if __name__ == '__main__':
    Fire(compute_avg_agg)