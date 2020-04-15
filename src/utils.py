import pickle
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
from glob import glob
import sys

def in_notebook():
    """
    Returns ``True`` if the module is running in IPython kernel,
    ``False`` if in IPython shell or other Python shell.
    """
    return 'ipykernel' in sys.modules

if in_notebook():
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm

def to_pickle(obj, fn):
    with open(fn, 'wb') as f:
        pickle.dump(obj, f)
def read_pickle(fn):
    with open(fn, 'rb') as f:
        return pickle.load(f)

def plot_hist(h, ax=None, ylim=None, name='', train=True, valid=True, **kwargs):
    if ax is None: fig, ax = plt.subplots()
    if train: ax.plot(h['loss'], label=f'{name}', **kwargs)
    if valid: ax.plot(h['val_loss'], **kwargs)
    ax.legend()
    if ylim is not None: ax.set_ylim(ylim)


def plot_losses(path, exp_ids, plot_lrs=True, ylim=None, log=False):
    exp_ids = [str(exp_id) for exp_id in exp_ids]
    fig, axs = plt.subplots(2 if plot_lrs else 1, 1, figsize=(15, 15 if plot_lrs else 7))
    colors = sns.palettes.color_palette(n_colors=len(exp_ids))
    for exp_id, c, in zip(exp_ids, colors):
        fn = glob(f'{path}{exp_id}*.pkl')[0]
        name = fn.split('/')[-1].split('_history.pkl')[0]
        h = read_pickle(fn)
        plot_hist(h, axs[0], name=name, valid=False, c=c, lw=2)
        plot_hist(h, axs[0], name=name, train=False, c=c, ls='--', lw=2)


        if plot_lrs:
            axs[1].plot(h['lr'], c=c)

    axs[0].set_ylim(ylim)
    if log: axs[0].set_yscale('log')