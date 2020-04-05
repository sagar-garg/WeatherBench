from pathlib import Path
import xarray as xr
from tqdm import tqdm
import fire

def interpolate_na(ds):
    try:
        a = ds.interpolate_na('lon', method='nearest', fill_value='extrapolate')
    except ValueError:
        # Sometimes there is only a single value at the pole which throws an error
        # in the iterpolation. This fixes it. But only for the lowest lat...
        v = list(ds.data_vars)[0]
        ds[v][:, :, 0] = ds[v].isel(lat=0).mean('lon').values[:, :, None]
        a = ds.interpolate_na('lon', method='nearest', fill_value='extrapolate')
    b = ds.interpolate_na('lat', method='nearest', fill_value='extrapolate')
    return xr.concat([a, b], 'tmp').mean('tmp')

def main(path):
    path = Path(path)
    path_int = Path(str(path) + '_int')
    for d in path.iterdir():
        print(d.name)
        (path_int / d.name).mkdir(exist_ok=True)
        for fn in tqdm(list(d.iterdir())):
            ds = xr.open_dataset(fn)
            interpolate_na(ds).to_netcdf(path_int / d.name / fn.name)

if __name__ == '__main__':
    fire.Fire(main)