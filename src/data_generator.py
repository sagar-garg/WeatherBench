import re
import numpy as np
import xarray as xr
import tensorflow as tf
import tensorflow.keras as keras
import datetime
from src.utils import *
import pandas as pd
import pdb
import logging
from tqdm import tqdm

def _tensor_feature(value):
    value = tf.io.serialize_tensor(value).numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def serialize_example(X, y):
    feature = {
        'X': _tensor_feature(X),
        'y': _tensor_feature(y)
    }
    feature = tf.train.Features(feature=feature)
    example_proto = tf.train.Example(features=feature)
    return example_proto.SerializeToString()

features = {
    'X': tf.io.FixedLenFeature([], tf.string),
    'y': tf.io.FixedLenFeature([], tf.string)
}

def _parse(example_proto):
    return tf.io.parse_single_example(example_proto, features)

def decode(example_proto):
    dic = _parse(example_proto)
    X = tf.io.parse_tensor(dic['X'], np.float32)
    y = tf.io.parse_tensor(dic['y'], np.float32)
    return X, y


class DataGenerator(keras.utils.Sequence):
    def __init__(self, ds, var_dict, lead_time, batch_size=32, shuffle=True, load=True,
                 mean=None, std=None, output_vars=None, data_subsample=1, norm_subsample=1,
                 nt_in=1, dt_in=1, cont_time=False, fixed_time=False, multi_dt=1, verbose=0,
                 min_lead_time=None, las_kernel=None, las_gauss_std=None, normalize=True,
                 tfrecord_files=None, tfr_buffer_size=1000, tfr_num_parallel_calls=1,
                 cont_dt=1, tfr_prefetch=None, tfr_repeat=True, y_roll=None, X_roll=None,
                 discard_first=None, tp_log=None, tfr_out=False, tfr_out_idxs=None,
                 old_const=False, is_categorical=False, num_bins=50, bin_min=-5, bin_max=5,
                 predict_difference=False, quantile_bins=None):
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
            data_subsample: Only take every ith time step
            norm_subsample: Same for normalization. This is AFTER data_subsample!
            nt_in: How many time steps for input. AFTER data_subsample!
            dt_in: Interval of input time steps. AFTER data_subsample!
        """
        if verbose: print('DG start', datetime.datetime.now().time())
        self.ds = ds
        self.var_dict = var_dict
        self.output_vars = output_vars
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.lead_time = lead_time
        self.nt_in = nt_in
        self.dt_in = dt_in
        self.cont_time = cont_time
        self.min_lead_time = min_lead_time
        self.fixed_time = fixed_time
        self.multi_dt = multi_dt
        self.tfrecord_files = tfrecord_files
        self.normalize = normalize
        self.tfr_num_parallel_calls = tfr_num_parallel_calls
        self.tfr_buffer_size = tfr_buffer_size
        self.cont_dt = cont_dt
        self.tfr_prefetch = tfr_prefetch
        self.tfr_repeat = tfr_repeat
        self.tfr_out = tfr_out
        self.y_roll = y_roll
        self.X_roll = X_roll
        self.tfr_max_lead = 120
        self.tfr_out_idxs = tfr_out_idxs
        self.old_const = old_const
        self.is_categorical = is_categorical
        self.num_bins = num_bins
        self.bin_min = bin_min
        self.bin_max = bin_max
        self.predict_difference = predict_difference
        if self.predict_difference:
            assert self.tfrecord_files is None, 'difference does not work for tfr'
        self.quantile_bins = quantile_bins

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
                da = ds[var]
                if tp_log and var == 'tp':
                    da = log_trans(da, tp_log)
                try:
                    data.append(da.sel(level=levels))
                    level_names += [f'{var}_{level}' for level in levels]
                except ValueError:
                    data.append(da.expand_dims({'level': generic_level}, 1))
                    level_names.append(var)

        self.data = xr.concat(data, 'level').transpose('time', 'lat', 'lon', 'level')
        if discard_first is not None:
            self.data = self.data.isel(time=slice(discard_first, None))
        self.data['level_names'] = xr.DataArray(
            level_names, dims=['level'], coords={'level': self.data.level})
        if output_vars is None:
            self.output_idxs = range(len(self.data.level))
        else:
            self.output_idxs = [i for i, l in enumerate(self.data.level_names.values)
                                if any([bool(re.match(o, l)) for o in output_vars])]
        self.const_idxs = [i for i, l in enumerate(self.data.level_names) if l in var_dict['constants']]
        self.not_const_idxs = [i for i, l in enumerate(self.data.level_names) if l not in var_dict['constants']]

        # Subsample
        self.data = self.data.isel(time=slice(0, None, data_subsample))
        self.raw_data = self.data
        self.dt = self.data.time.diff('time')[0].values / np.timedelta64(1, 'h')
        self.dt_in = int(self.dt_in // self.dt)
        self.nt_offset = (nt_in - 1) * self.dt_in

        if self.min_lead_time is None:
            self.min_nt = 1
        else:
            self.min_nt = int(self.min_lead_time / self.dt)

        # Normalize
        if verbose: print('DG normalize', datetime.datetime.now().time())
        if mean is not None:
            self.mean = mean
        else:
            self.mean = self.data.isel(time=slice(0, None, norm_subsample)).mean(
                ('time', 'lat', 'lon')).compute()
            if 'tp' in self.data.level_names:  # set tp mean to zero but not if ext
                tp_idx = list(self.data.level_names).index('tp')
                self.mean.values[tp_idx] = 0

        if std is not None:
            self.std = std
        else:
            self.std = self.data.isel(time=slice(0, None, norm_subsample)).std(
                ('time', 'lat', 'lon')).compute()

        if tp_log is not None:
            self.mean.attrs['tp_log'] = tp_log
            self.std.attrs['tp_log'] = tp_log
        if normalize:
            self.data = (self.data - self.mean) / self.std

        if verbose: print('DG load', datetime.datetime.now().time())
        if load:
            if verbose: print('Loading data into RAM')
            self.data.load()
        if verbose: print('DG done', datetime.datetime.now().time())

        if self.X_roll is not None:
            self.X_roll = int(self.X_roll // self.dt)
            self.X_rolled = self.data.rolling(time=self.X_roll).mean()
            self.nt_offset += self.X_roll

        self.on_epoch_end()

        if self.y_roll is not None:
            self.y_roll = int(self.y_roll // self.dt)
            assert self.y_roll < self.nt, 'nt must be larger than y_roll'
            self.y_rolled = self.data.isel(level=self.output_idxs).rolling(time=self.y_roll).mean()

        if self.tfrecord_files is not None:
            self.is_tfr = True
            self._setup_tfrecord_ds()
        else:
            self.is_tfr = False
            self.tfr_dataset = None

        if self.is_categorical:
            if self.quantile_bins:
                # Get a sample
                self.is_categorical = False
                _, y_sample = self._get_item(0)   # Assume shuffled
                self.is_categorical = True

                if 'tp' in self.output_vars:
                    assert len(self.output_vars) == 1, 'tp must be stand-alone'
                    y_sample = y_sample.flatten()[y_sample.flat > 0]
                
                self.bins = np.quantile(y_sample.flatten(), np.linspace(0, 1, self.num_bins+1))
                self.bins[0] = -np.inf; self.bins[-1] = np.inf
            else:
                self.bins = np.linspace(self.bin_min, self.bin_max, self.num_bins+1)
                self.bins[0] = -np.inf; self.bins[-1] = np.inf  # for rare out-of-bound cases.

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.idxs = np.arange(self.nt_offset, self.n_samples)
        if self.shuffle:
            np.random.shuffle(self.idxs)

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.ceil(len(self.idxs) / self.batch_size))
    
    @property
    def shape(self):
        return (
            len(self.data.lat),
            len(self.data.lon),
            len(self.data.level.isel(level=self.not_const_idxs)) * self.nt_in + len(self.data.level.isel(
                level=self.const_idxs)) + self.cont_time
        )

    @property
    def nt(self):
        assert (self.lead_time / self.dt).is_integer(), "lead_time and dt not compatible."
        return int(self.lead_time / self.dt)

    @property
    def init_time(self):
        stop = -self.nt
        if self.is_tfr:
            stop += -(self.tfr_max_lead - self.lead_time // self.dt)
        return self.data.isel(time=slice(self.nt_offset, int(stop))).time

    @property
    def valid_time(self):
        start = self.nt+self.nt_offset
        stop = None
        if self.multi_dt > 1:
            diff = self.nt - self.nt // self.multi_dt
            start -= diff; stop = -diff
        if self.is_tfr:
            stop = -int((self.tfr_max_lead - self.lead_time) // self.dt)
            if stop == 0:
                stop = None
        return self.data.isel(time=slice(start, stop)).time

    @property
    def n_samples(self):
        return self.data.isel(time=slice(0, -self.nt)).shape[0]

    def __getitem__(self, i):
        if self.tfrecord_files is None:
            if hasattr(self, 'cheat'):
                X, y = self._get_item(i)
                return X, y[-1]
            else:
                return self._get_item(i)
        else:
            return self._get_tfrecord_item(i)

    def _get_item(self, i):
        'Generate one batch of data'
        idxs = self.idxs[i * self.batch_size:(i + 1) * self.batch_size]

        if self.cont_time:
            if not self.fixed_time:
                nt = np.random.randint(self.min_nt, self.nt + 1, len(idxs))
            else:
                nt = np.ones(len(idxs), dtype='int') * self.nt
            ftime = (nt * self.dt / 100)[:, None, None] * np.ones((1, len(self.data.lat),
                                                                   len(self.data.lon)))
        else:
            nt = self.nt

        if self.X_roll is not None:
            X_data = self.X_rolled
        else:
            X_data = self.data

        X = X_data.isel(time=idxs).values.astype('float32')

        if self.multi_dt > 1: consts = X[..., self.const_idxs]

        if self.nt_in > 1:
            if self.old_const:
                X = np.concatenate([
                                       self.data.isel(time=idxs - nt_in * self.dt_in).values
                                       for nt_in in range(self.nt_in - 1, 0, -1)
                                   ] + [X], axis=-1).astype('float32')
            else:
                X = np.concatenate([
                                       X_data.isel(time=idxs - nt_in * self.dt_in).values[..., self.not_const_idxs]
                                       for nt_in in range(self.nt_in - 1, 0, -1)
                                   ] + [X], axis=-1).astype('float32')

        if self.multi_dt > 1:
            X = [X[..., self.not_const_idxs], consts]
            step = self.nt // self.multi_dt
            y = [
                self.data.isel(time=idxs + nt, level=self.output_idxs).values.astype('float32')
                for nt in np.arange(step, self.nt + step, step)
            ]
        elif self.y_roll is not None:
            y = self.y_rolled.isel(
                time=idxs + nt,
            ).values.astype('float32')
        elif self.tfr_out:
            assert self.batch_size == 1, 'bs must be one'
            time_slice = slice(idxs[0]+self.min_nt, idxs[0]+self.nt+1)
            y = self.data.isel(time=time_slice, level=self.output_idxs).values.astype('float32')[None]
        elif self.predict_difference:
            y = (
                self.data.isel(time=idxs + nt, level=self.output_idxs).values -
                self.data.isel(time=idxs, level=self.output_idxs).values
            ).astype('float32')
        else:
            y = self.data.isel(time=idxs + nt, level=self.output_idxs).values.astype('float32')

        if self.is_categorical:
            y_shape = y.shape
            y = pd.cut(y.reshape(-1), self.bins, labels=False).reshape(y_shape)
            y = tf.keras.utils.to_categorical(y, num_classes=self.num_bins)
            y = y.reshape((*y_shape, self.num_bins))

        if self.cont_time:
            X = np.concatenate([X, ftime[..., None]], -1).astype('float32')
        return X, y


    def _decode(self, example_proto):
        dic = _parse(example_proto)
        X = tf.io.parse_tensor(dic['X'], np.float32)
        y = tf.io.parse_tensor(dic['y'], np.float32)
        if self.tfr_out_idxs is not None:
            y = tf.gather(y, self.tfr_out_idxs, axis=-1)
        if self.cont_time:
            if self.fixed_time:
                y_idx = self.nt-1
            else:
                y_idx = tf.random.uniform((), self.min_nt-1, self.nt, dtype=tf.int32)
            y_time = (y_idx+1) * self.dt
            ftime = (y_time / 100) * np.ones((len(self.data.lat), len(self.data.lon), 1))
            X = tf.concat([X, tf.cast(ftime, tf.float32)], -1)
            return X, y[y_idx]
        else:
            y_idx = self.nt-1
            return X, y[y_idx]

    def _setup_tfrecord_ds(self):
        # Find all files to be used
        if type(self.tfrecord_files) is list:
            tfr_fns = self.tfrecord_files
        else:
            tfr_fns = sorted(glob(self.tfrecord_files))

        dataset = tf.data.TFRecordDataset(
            tfr_fns, num_parallel_reads=self.tfr_num_parallel_calls
        ).map(self._decode)

        if self.shuffle:
            dataset = dataset.shuffle(
                buffer_size=self.tfr_buffer_size, reshuffle_each_iteration=True
            )

        self.tfr_dataset = dataset.batch(self.batch_size)
        # if self.tfr_repeat:
        #     self.tfr_dataset = self.tfr_dataset.repeat()
        if self.tfr_prefetch is not None:
            self.tfr_dataset = self.tfr_dataset.prefetch(self.tfr_prefetch)
        self.tfr_dataset_np = self.tfr_dataset.as_numpy_iterator()


    def _get_tfrecord_item(self, i):
        X, y = next(self.tfr_dataset_np)
        return X, y

    def to_tfr(self, savedir, steps_per_file=250):
        assert self.batch_size == 1, 'bs must be one'
        for i, (X, y) in tqdm(enumerate(self)):
            if i % steps_per_file == 0:
                c = int(np.floor(i / steps_per_file))
                fn = f'{savedir}/{str(c).zfill(3)}.tfrecord'
                print('Writing to file:', fn)
                writer = tf.io.TFRecordWriter(fn)
            serialized_example = serialize_example(X[0], y[0])  # Remove batch dimension
            writer.write(serialized_example)
            if i + 1 % steps_per_file == 0:
                writer.close()
        writer.close()



class CombinedDataGenerator(keras.utils.Sequence):

    def __init__(self, dgs, batch_size):
        self.dgs = dgs
        self.lens = np.array([len(dg.idxs) for dg in self.dgs])
        self.data = self.dgs[0].data
        self.batch_size = batch_size
        self.bss = np.round(self.lens / self.lens.sum() * batch_size)
        missing = batch_size - self.bss.sum()
        self.bss[0] += missing
        assert self.bss.sum() == batch_size, 'Batch sizes dont add up'
        print('Individual batch sizes:', self.bss)
        for dg, bs in zip(dgs, self.bss): dg.batch_size = int(bs)
        self.mean = self.dgs[0].mean
        self.std = self.dgs[0].std
        self.tfr_dataset = self.dgs[0].tfr_dataset

    @property
    def shape (self):
        return self.dgs[0].shape

    def __len__(self):
        total_samples = np.sum([len(dg.idxs) for dg in self.dgs])
        return int(np.ceil(total_samples / self.batch_size))

    def __getitem__(self, i):
        Xs = []
        ys = []
        for dg in self.dgs:
            X, y = dg[i]
            Xs.append(X)
            ys.append(y)
        return np.concatenate(Xs), np.concatenate(ys)

    def on_epoch_end(self):
        for dg in self.dgs:
            dg.on_epoch_end()


def create_predictions(model, dg, multi_dt=False, parametric=False, verbose=0, no_mean=False):
    """Create non-iterative predictions"""
    level_names = dg.data.isel(level=dg.output_idxs).level_names
    level = dg.data.isel(level=dg.output_idxs).level
    if parametric:
        # pdb.set_trace()
        lvl = level_names.values
        mm, ss = [], []
        for l in lvl:
            m = l.split('_'); s = l.split('_')
            m[0] += '-mean'; s[0] += '-std'
            mm.append('_'.join(m)); ss.append('_'.join(s))
        lvl = mm + ss
        level_names = xr.concat([level_names]*2, dim='level')
        level_names[:] = lvl
        level = xr.concat([level]*2, dim='level')

    

    mean = dg.mean.isel(level=dg.output_idxs).values if not no_mean else 0
    std = dg.std.isel(level=dg.output_idxs).values

    if dg.is_categorical:
        # preds = model.predict(dg.tfr_dataset or dg, verbose=verbose)
        preds = []
        for X, y in tqdm(dg):
            preds.append(model.predict(X))
        preds = np.concatenate(preds)
        if dg.predict_difference:
            unnormalized_bins = dg.bins * std[:, None]
        else:
            unnormalized_bins = dg.bins * std[:, None] + mean[:, None] 
        
        level_names = dg.data.isel(level=dg.output_idxs).level_names
        level = dg.data.isel(level=dg.output_idxs).level
        preds = xr.DataArray(
            preds,
            dims=['time', 'lat', 'lon', 'level', 'bin'],
            coords={'time': dg.valid_time, 'lat': dg.data.lat, 'lon': dg.data.lon,
                    'level': level,
                    'level_names': level_names,
                    'bin': np.arange(dg.num_bins),
                    },
        )
    else:
        preds = model.predict(dg.tfr_dataset or dg, verbose=verbose)
        preds = xr.DataArray(
            preds[0] if multi_dt else preds,
            dims=['time', 'lat', 'lon', 'level'],
            coords={'time': dg.valid_time, 'lat': dg.data.lat, 'lon': dg.data.lon,
                    'level': level,
                    'level_names': level_names
                    },
        )
        # Unnormalize
        
        if parametric:
            mean = np.concatenate([mean, np.zeros_like(mean)])
            std = np.concatenate([std]*2)
        preds = preds * std + mean

        # Reverse tranforms
        if hasattr(dg.mean, 'tp_log') and 'tp' in unique_vars:
            tp_idx = list(preds.level_names).index('tp')
            preds.values[..., tp_idx] = log_retrans(preds.values[..., tp_idx], dg.mean.tp_log)

    unique_vars = list(set([l.split('_')[0] for l in preds.level_names.values]))

    das = []
    for v in unique_vars:
        idxs = [i for i, vv in enumerate(preds.level_names.values) if vv.split('_')[0] == v]
        da = preds.isel(level=idxs).squeeze().drop('level_names')
        if dg.is_categorical:
            bin_edges = unnormalized_bins[idxs].squeeze()
            da.attrs['bin_edges'] = bin_edges
            da.attrs['mid_points'] = (bin_edges[1:] + bin_edges[:-1]) / 2
            da.attrs['bin_width'] = bin_edges[1] - bin_edges[0]
        if not 'level' in da.dims: da = da.drop('level')
        das.append({v: da})
    return xr.merge(das)

def create_cont_predictions(model, dg, max_lead_time=120, dt=12, lead_time=None):
    dg.fixed_time = True
    lead_time = np.arange(dt, max_lead_time+dt, dt) if lead_time is None else lead_time
    lead_time = xr.DataArray(lead_time, dims={'lead_time': lead_time}, name='lead_time')
    preds = []
    for l in tqdm(lead_time):
        dg.lead_time = l.values; dg.on_epoch_end()
        if dg.is_tfr: dg._setup_tfrecord_ds()
        p = create_predictions(model, dg)
        p['time'] = dg.init_time
        preds.append(p)
    return xr.concat(preds, lead_time)

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