from src.utils import *
from src.data_generator import *
from configargparse import ArgParser
import ast, os

def convert_to_tfr(savedir, var_dict, datadir, years, cmip=False, cmip_dir=None, compute_norm=True,
                   norm_subsample=30000, data_subsample=1):
    os.makedirs(savedir, exist_ok=True)
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
    ds = ds.sel(time=slice(*years))
    dg = DataGenerator(
        ds, var_dict, lead_time=1, batch_size=1,
        data_subsample=data_subsample, norm_subsample=norm_subsample,
        mean=None if compute_norm else '', std=None if compute_norm else '',
        normalize=False, shuffle=False, load=True, verbose=True
    )
    dg.to_tfrecord(savedir)
    if compute_norm:
        dg.mean.to_netcdf(f'{savedir}mean.nc')
        dg.std.to_netcdf(f'{savedir}std.nc')

if __name__ == '__main__':
    p = ArgParser()
    p.add_argument('-c', '--my-config', is_config_file=True, help='config file path')
    p.add_argument('--savedir', type=str, required=True, help='Where to save TFR files')
    p.add_argument('--datadir', type=str, required=True, help='Path to data')
    p.add_argument('--var_dict', required=True, help='Variables: as an ugly dictionary...')
    p.add_argument('--years', type=str, nargs='+', default=('1979', '2018'), help='Start/stop years')
    p.add_argument('--cmip', type=int, default=0, help='Is CMIP')
    p.add_argument('--cmip_dir', type=str, default=None, nargs='+', help='Dirs for CMIP data')
    p.add_argument('--compute_norm', type=int, default=1, help='Is CMIP')
    p.add_argument('--data_subsample', type=int, default=1, help='Subsampling for training data')
    p.add_argument('--norm_subsample', type=int, default=1, help='Subsampling for mean/std')
    args = p.parse_args()
    args.var_dict = ast.literal_eval(args.var_dict)
    convert_to_tfr(args.savedir, args.var_dict, args.datadir, args.years, args.cmip, args.cmip_dir,
                   args.compute_norm, args.norm_subsample, args.data_subsample)

