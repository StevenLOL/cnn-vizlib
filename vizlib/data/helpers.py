#!/usr/bin/env python2.7
# encoding: utf-8
'''
'''
import numpy as np
import matplotlib.pyplot as plt
import theano

class DataSet(object):

    def __init__(self, X, y):
        self.X = ensure_batch_chan_row_col(X).astype(theano.config.floatX)
        # TODO: support regression datasets
        self.y = np.array(y).astype(np.int32)

        self.mean = None
        self.std = None

    def standardize(self):
        self.X, self.mean, self.std = standardize(self.X)
        return self

    def __getitem__(self, key):
        return DataSet(self.X[key], self.y[key])

    def __len__(self):
        return len(self.X)

    def unstandardize(self):
        if self.mean is None or self.std is None:
            raise ValueError('Data is not standardized')
        self.X = unstandardize(self.X, self.mean, self.std)
        return self

    def shuffle(self):
        perm = np.arange(len(self.X))
        np.random.shuffle(perm)
        self.X = self.X[perm]
        self.y = self.y[perm]
        return self

    def show_sample(self):
        n_samples_per_class = 4
        classes = sorted(set(self.y))
        n_classes = len(classes)

        fig, axes = plt.subplots(n_classes, n_samples_per_class,
                                subplot_kw=dict(xticks=[], yticks=[]))

        i = 0
        for sub_axes, c in zip(axes, classes):
            samples = self.X[self.y == c][:n_samples_per_class]
            # might not always have n_samples_per_class, i_inner fixes this.
            for ax, x in zip(sub_axes, samples):
                ax.imshow(x.squeeze())
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
    mean = standardized.mean()
    standardized -= mean
    std = standardized.std()
    standardized /= std
    return (standardized, mean, std)

@allow_inplace
def unstandardize(standardized, mean, std):
    '''returns dataset'''
    dataset = standardized
    dataset *= std
    dataset += mean
    return dataset
