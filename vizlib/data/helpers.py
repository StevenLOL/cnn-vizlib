#!/usr/bin/env python2.7
# encoding: utf-8
'''
'''
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import theano

class DataSet(object):
    '''This dataset object is only for image data, which we assume to be
    BxCxNxM'''

    def __init__(self, X, y):
        self.X = ensure_batch_chan_row_col(X).astype(theano.config.floatX)
        # TODO: support regression datasets
        self.y = np.array(y).astype(np.int32)

        self.mean = None
        self.std = None
        self.unstandardize_func = None

    def standardize(self, standardization_type='individual'):
        if self.unstandardize_func is not None:
            self.unstandardize()

        if standardization_type == 'individual':
            self.X, self.mean, self.std = standardize(self.X)
            self.unstandardize_func = unstandardize

        elif standardization_type == 'global':
            self.X, self.mean, self.std = standardize_globally(self.X)
            self.unstandardize_func = unstandardize

        elif standardization_type == 'class':
            raise NotImplemented()

        else:
            raise ValueError()

        self.standardization_type = standardization_type
        return self

    def one_of_class(self):
        y = self.classes
        X = np.array([self.X[self.y == c][0] for c in y])
        return DataSet(X, y)

    @property
    def classes(self):
        return list(set(list(self.y)))

    def __getitem__(self, key):
        return DataSet(self.X[key], self.y[key])

    def __len__(self):
        return len(self.X)

    def unstandardize(self):
        if self.unstandardize_func is None:
            raise ValueError('Data is not standardized')
        self.X = self.unstandardize_func(self.X, self.mean, self.std)
        self.unstandardize_func = None
        return self

    def shuffle(self):
        perm = np.arange(len(self.X))
        np.random.shuffle(perm)
        self.X = self.X[perm]
        self.y = self.y[perm]
        return self

    def show_sample(self):
        n_samples_per_class = min(4, pd.value_counts(self.y).max())
        classes = sorted(set(self.y))
        n_classes = len(classes)
        vmin, vmax = self.X.min(), self.X.max()

        fig, axes = plt.subplots(n_classes, n_samples_per_class,
                                subplot_kw=dict(xticks=[], yticks=[]))

        i = 0
        for sub_axes, c in zip(axes, classes):
            samples = self.X[self.y == c][:n_samples_per_class]
            if n_samples_per_class == 1:
                sub_axes = [sub_axes]
            for ax, x in zip(sub_axes, samples):
                ax.matshow(x.squeeze(), vmin=vmin, vmax=vmax)
                ax.axis('off')
                ax.set_title(c)
            i += n_samples_per_class
        return self

def ensure_batch_chan_row_col(X):
    X = np.array(X)
    if len(X.shape) > 4:
        raise ValueError()
    if len(X.shape) == 4:
        return X
    if len(X.shape) == 3:
        return X.reshape(X.shape[0], 1, X.shape[1], X.shape[2])
    raise ValueError()

def allow_inplace(f):
    '''add inplace argument w/ default value True'''
    def w_implace(*args, **kwargs):
        inplace = 'inplace' not in kwargs or kwargs['inplace']
        if not inplace:
            args[0] = args[0].copy()
        return f(*args, **kwargs)
    return w_implace

def plot_imgs(X):
    X = np.array(X)
    n = np.ceil(np.sqrt(len(X)))
    vmin = X.min()
    vmax = X.max()
    for i, (x) in enumerate(X, 1):
        plt.subplot(n, n, i)
        plt.imshow(x.squeeze(), vmin=vmin, vmax=vmax)
        plt.axis('off')

@allow_inplace
def standardize(dataset):
    '''returns (dataset, mean, std)'''

    # want 0 mean, 1 std, so first center on the mean:
    standardized = dataset
    mean = standardized.mean(axis=(1, 2, 3))[:, None, None, None]
    standardized -= mean
    std = standardized.std(axis=(1, 2, 3))[:, None, None, None]
    standardized /= std
    return (standardized, mean, std)

@allow_inplace
def unstandardize(standardized, mean, std):
    '''returns dataset'''
    dataset = standardized
    dataset *= std
    dataset += mean
    return dataset

@allow_inplace
def standardize_globally(dataset):
    standardized = dataset
    mean = standardized.mean()
    standardized -= mean
    std = standardized.std()
    standardized /= std
    return (standardized, mean, std)
