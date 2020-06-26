from src.utils import *
from src.train import *
from src.data_generator import *
from configargparse import ArgParser
import ast, os
from fire import Fire

def convert_to_tfr(my_config, savedir='/data/stephan/WeatherBench/TFR', steps_per_file=250):
    args = load_args(my_config)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args['gpu'])

    print('Load data generators')
    if args['cmip']: args['cmip_dir'] = args['cmip_dir'][0]
    dg_train, dg_valid, dg_test = load_data(**args)

    savedir = savedir + '/' + args['exp_id']
    os.makedirs(savedir + '/train', exist_ok=True)
    os.makedirs(savedir + '/valid', exist_ok=True)
    os.makedirs(savedir + '/test', exist_ok=True)

    print('Save TFR files')
    dg_train.to_tfr(savedir + '/train', steps_per_file)
    dg_valid.to_tfr(savedir + '/valid', steps_per_file)
    dg_test.to_tfr(savedir + '/test', steps_per_file)

    print('Save norm files')
    dg_train.mean.to_netcdf(f'{savedir}/mean.nc')
    dg_train.std.to_netcdf(f'{savedir}/std.nc')

if __name__ == '__main__':
    Fire(convert_to_tfr)

