import re
import numpy as np
import xarray as xr
import tensorflow.keras as keras
import datetime
import pdb

class DataGenerator(keras.utils.Sequence):
    def __init__(self, ds, var_dict, lead_time, batch_size=32, shuffle=True, load=True,
                 mean=None, std=None, output_vars=None, data_subsample=1, norm_subsample=1,
                 nt_in=1, dt_in=1):
        """
        Data generator for WeatherBench data.
        Template from https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
        Args:
            ds: Dataset containing all variables
            var_dict: Dictionary of the form {'var': level}. Use None for level if data is of single level
            lead_time: Lead time in hours
            batch_size: Batch size
            shuffle: bool. If True, data is shuffled.
            load: bool. If True, datadet is loaded into RAM.
            mean: If None, compute mean from data.
            std: If None, compute standard deviation from data.
        """
        print('DG start', datetime.datetime.now().time())
        self.ds = ds
        self.var_dict = var_dict
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.lead_time = lead_time
        self.nt_in = nt_in
        self.dt_in = dt_in
        self.nt_offset = (nt_in * dt_in) - 1

        data = []
        level_names = []
        generic_level = xr.DataArray([1], coords={'level': [1]}, dims=['level'])
        for long_var, params in var_dict.items():
            if long_var == 'constants':
                for var in params:
                    data.append(ds[var].expand_dims(
                        {'level': generic_level, 'time': ds.time}, (1, 0)
                    ))
                    level_names.append(var)
            else:
                var, levels = params
                try:
                    data.append(ds[var].sel(level=levels))
                    level_names += [f'{var}_{level}' for level in levels]
                except ValueError:
                    data.append(ds[var].expand_dims({'level': generic_level}, 1))
                    level_names.append(var)

        self.data = xr.concat(data, 'level').transpose('time', 'lat', 'lon', 'level')
        self.data['level_names'] = xr.DataArray(
            level_names, dims=['level'], coords={'level': self.data.level})
        if output_vars is None:
            self.output_idxs = range(len(self.data.level))
        else:
            self.output_idxs = [i for i, l in enumerate(self.data.level_names.values)
                                if any([bool(re.match(o, l)) for o in output_vars])]

        # Subsample
        self.data = self.data.isel(time=slice(0, None, data_subsample))
        self.dt = self.data.time.diff('time')[0].values / np.timedelta64(1, 'h')
        assert (self.lead_time / self.dt).is_integer(), "lead_time and dt not compatible."
        self.nt = int(self.lead_time / self.dt)

        # Normalize
        print('DG normalize', datetime.datetime.now().time())
        self.mean = self.data.isel(time=slice(0, None, norm_subsample)).mean(
            ('time', 'lat', 'lon')).compute() if mean is None else mean
        #         self.std = self.data.std('time').mean(('lat', 'lon')).compute() if std is None else std
        self.std = self.data.isel(time=slice(0, None, norm_subsample)).std(
            ('time', 'lat', 'lon')).compute() if std is None else std
        self.data = (self.data - self.mean) / self.std

        self.n_samples = self.data.isel(time=slice(0, -self.nt)).shape[0]
        self.init_time = self.data.isel(time=slice(self.nt_offset, -self.nt)).time
        self.valid_time = self.data.isel(time=slice(self.nt+self.nt_offset, None)).time

        self.on_epoch_end()

        # For some weird reason calling .load() earlier messes up the mean and std computations
        print('DG load', datetime.datetime.now().time())
        if load: print('Loading data into RAM'); self.data.load()
        print('DG done', datetime.datetime.now().time())

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.ceil(len(self.idxs) / self.batch_size))

    def __getitem__(self, i):
        'Generate one batch of data'
        idxs = self.idxs[i * self.batch_size:(i + 1) * self.batch_size]
        X = self.data.isel(time=idxs).values
        if self.nt_in > 1:
            X = np.concatenate([
                self.data.isel(time=idxs-nt_in*self.dt_in).values for nt_in in range(self.nt_in-1, 0, -1)
            ] + [X], axis=-1)
        y = self.data.isel(time=idxs + self.nt, level=self.output_idxs).values
        return X.astype('float32'), y.astype('float32')

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.idxs = np.arange(self.nt_offset, self.n_samples)
        if self.shuffle:
            np.random.shuffle(self.idxs)


def create_predictions(model, dg):
    """Create non-iterative predictions"""
    preds = xr.DataArray(
        model.predict(dg),
        dims=['time', 'lat', 'lon', 'level'],
        coords={'time': dg.valid_time, 'lat': dg.data.lat, 'lon': dg.data.lon,
                'level': dg.data.isel(level=dg.output_idxs).level,
                'level_names': dg.data.isel(level=dg.output_idxs).level_names
                },
    )
    # Unnormalize
    preds = (preds * dg.std.isel(level=dg.output_idxs).values +
             dg.mean.isel(level=dg.output_idxs).values)
    unique_vars = list(set([l.split('_')[0] for l in preds.level_names.values]))

    das = []
    for v in unique_vars:
        idxs = [i for i, vv in enumerate(preds.level_names.values) if vv.split('_')[0] in v]
        da = preds.isel(level=idxs).squeeze().drop('level_names')
        if not 'level' in da.dims: da = da.drop('level')
        das.append({v: da})
    return xr.merge(das)

# TODO: Outdated
# def create_iterative_predictions(model, dg, max_lead_time=5 * 24):
#     """Create iterative predictions"""
#     state = dg.data[:dg.n_samples]
#     preds = []
#     for _ in range(max_lead_time // dg.lead_time):
#         state = model.predict(state)
#         p = state * dg.std.values + dg.mean.values
#         preds.append(p)
#     preds = np.array(preds)
#
#     lead_time = np.arange(dg.lead_time, max_lead_time + dg.lead_time, dg.lead_time)
#     das = []
#     lev_idx = 0
#     for var, levels in dg.var_dict.items():
#         if levels is None:
#             das.append(xr.DataArray(
#                 preds[:, :, :, :, lev_idx],
#                 dims=['lead_time', 'time', 'lat', 'lon'],
#                 coords={'lead_time': lead_time, 'time': dg.init_time, 'lat': dg.ds.lat, 'lon': dg.ds.lon},
#                 name=var
#             ))
#             lev_idx += 1
#         else:
#             nlevs = len(levels)
#             das.append(xr.DataArray(
#                 preds[:, :, :, :, lev_idx:lev_idx + nlevs],
#                 dims=['lead_time', 'time', 'lat', 'lon', 'level'],
#                 coords={'lead_time': lead_time, 'time': dg.init_time, 'lat': dg.ds.lat, 'lon': dg.ds.lon,
#                         'level': levels},
#                 name=var
#             ))
#             lev_idx += nlevs
#     return xr.merge(das)