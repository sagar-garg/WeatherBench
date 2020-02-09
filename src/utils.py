import pickle
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt

def to_pickle(obj, fn):
    with open(fn, 'wb') as f:
        pickle.dump(obj, f)
def read_pickle(fn):
    with open(fn, 'rb') as f:
        return pickle.load(f)

def plot_hist(h, ax=None, ylim=None, name='', train=True, valid=True, **kwargs):
    if ax is None: fig, ax = plt.subplots()
    if train: ax.plot(h['loss'], label=f'{name}train', **kwargs)
    if valid: ax.plot(h['val_loss'], label=f'{name}valid', **kwargs)
    ax.legend()
    if ylim is not None: ax.set_ylim(ylim)
