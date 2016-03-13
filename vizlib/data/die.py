import numpy as np
import os

from helpers import DataSet

DIE_X_FILE = os.path.join(os.path.dirname(__file__), 'die.X.npy')
DIE_Y_FILE = os.path.join(os.path.dirname(__file__), 'die.y.improved.npy')

def die(standardization_type='individual'):
    X = np.load(DIE_X_FILE)
    # File is numbered from 1 to 6 (inclusive), and as strings
    y = np.load(DIE_Y_FILE).astype(np.int32) - 1
    return DataSet(X, y, label_lookup=dict(zip(y, y + 1)))\
            .to_grayscale()\
            .standardize(standardization_type)\
            .to_nxmxc()\
            .shuffle()

def cache(func):
    def wrapper():
        fname = os.path.join(os.path.dirname(__file__), func.__name__ + '.p')
        try:
            ds = DataSet.from_pickle(fname)
        except:
            ds = func()
            ds.to_pickle(fname)
        return ds
    return wrapper

def die_zoomed(standardization_type, zoom_factor):
    ds = die(None)
    ds = ds.zoom(zoom_factor)
    ds.standardize(standardization_type)
    return ds


@cache
def die16(standardization_type='individual'):
    return die_zoomed(standardization_type, 1 / 8.0)

@cache
def die32(standardization_type='individual'):
    return die_zoomed(standardization_type, 1 / 4.0)

@cache
def die64(standardization_type='individual'):
    return die_zoomed(standardization_type, 1 / 2.0)
