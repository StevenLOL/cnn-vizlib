'''
'''
from __future__ import print_function

import numpy as np
import pandas as pd

import scipy.ndimage
import cv2

from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt
import theano

import pickle

class DataSet(object):
    '''This dataset object is only for image data, which we assume to be
    BxCxNxM'''

    def __init__(self, X, y, label_lookup=None):
        self.X = ensure_batch_chan_row_col(X).astype(theano.config.floatX)
        # TODO: support regression datasets
        self.y = np.array(y).astype(np.int32)

        self.mean = None
        self.std = None
        self.unstandardize_func = None
        self.standardization_type = None
        if label_lookup is None:
            self.label_lookup = dict(zip(y, y))
        else:
            self.label_lookup = label_lookup

    def to_pickle(self, fname):
        # FIXME: could not pickle the standardization functions.
        with open(fname, 'wb') as fh:
            state = (
                self.X,
                self.y,
                self.label_lookup
            )
            pickle.dump(state, fh, -1)

    @staticmethod
    def from_pickle(fname):
        with open(fname, 'rb') as fh:
            X, y, label_lookup = pickle.load(fh)
        result = DataSet(X, y, label_lookup)
        return result

    def standardize(self, standardization_type='individual'):
        types = ('individual', 'global', 'class', None)

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

        elif standardization_type is None:
            pass

        else:
            raise ValueError(
                '"{}" is not a valid standardization_type, expected one of: {}'\
                .format(standardization_type, types)
            )

        self.standardization_type = standardization_type
        return self

    def to_grayscale(self):
        standardization_type = self.standardization_type
        self.unstandardize()
        rgb = self.to_nxmxc().X
        gray = np.array([cv2.cvtColor(x, cv2.COLOR_RGB2GRAY)
                         for x in rgb])[:, None, :, :]
        self.standardize(standardization_type)
        return DataSet(gray, self.y)

    def one_of_class(self):
        y = self.classes
        X = np.array([self.X[self.y == c][0] for c in y])
        return DataSet(X, y, label_lookup=self.label_lookup)

    def to_nxmxc(self):
        return DataSet(np.rollaxis(self.X, 1, 4), self.y, label_lookup=self.label_lookup)

    def zoom(self, factor):
        # Depending on the shape, we need to perform certain kinds of zooming.
        # Unfortunately, we can not completely determine what configuration
        # we are in.
        # If a shape is (x, 3, x, 3) it is ambiguous
        # if we are in rgb space or not.
        is_rgb = self.X.shape[-1] == 3 and self.X.shape[1] != 3
        is_grayscale = self.X.shape[-1] == 1
        if is_rgb or is_grayscale:
            zoom_factors = (1, factor, factor, 1)
        else:
            zoom_factors = (1, 1, factor, factor)
        return DataSet(
            scipy.ndimage.zoom(self.X, zoom_factors),
            self.y,
            label_lookup=self.label_lookup
        )

    def show_misclassifications(self, y_pred, n_errors_per_class=None):
        classes = self.classes
        n_classes = len(classes)
        if n_errors_per_class is None:
            n_errors_per_class = 2 * n_classes
        to_show = number_of_errors_to_show(
            confusion_matrix(self.y, y_pred),
            n_errors_per_class
        )

        fig, axes = plt.subplots(
            n_classes,
            n_errors_per_class,
            subplot_kw=dict(xticks=[], yticks=[])
        )

        imgs = self.to_nxmxc().X

        for actual, sub_axes in enumerate(axes):
            ax_iter = iter(sub_axes)
            for pred in classes:
                # Show those errors where class actual is misclassified as pred
                for x in imgs[(y_pred == pred) & (self.y == actual)][:to_show[actual, pred]]:
                    ax = next(ax_iter)
                    ax.imshow(x.squeeze())
                    ax.set_title('{} as {}'.format(self.label_lookup[actual], self.label_lookup[pred]))
        return self

    @property
    def shape(self):
        return self.X.shape

    @property
    def classes(self):
        return sorted(list(set(list(self.y))))

    def __getitem__(self, key):
        return DataSet(self.X[key], self.y[key], label_lookup=self.label_lookup)

    def __len__(self):
        return len(self.X)

    def unstandardize(self):
        if self.unstandardize_func is None:
            return self
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
        nxmxc = self.to_nxmxc()
        for sub_axes, c in zip(axes, classes):
            samples = nxmxc.X[self.y == c][:n_samples_per_class]
            if n_samples_per_class == 1:
                sub_axes = [sub_axes]
            for ax, x in zip(sub_axes, samples):
                ax.imshow(x.squeeze(), vmin=vmin, vmax=vmax)
                ax.axis('off')
                ax.set_title(self.label_lookup[c])
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

def remove_diag(a):
    k = 0
    idxs = []
    for i in range(len(a)):
        for j in range(len(a)):
            if i != j:
                idxs.append(k)
            k += 1
    return a.flatten()[idxs].reshape((len(a), len(a) - 1))

def number_of_errors_to_show(conf_mat, total_errors_per_class):
    result = np.ones(conf_mat.shape, dtype=np.int)
    np.fill_diagonal(result, 0)
    result[conf_mat == 0] = 0
    errors_to_assign = total_errors_per_class - result.sum(axis=1)

    conf_mat = conf_mat.copy().astype(float)
    np.fill_diagonal(conf_mat, 0)
    q = remove_diag(conf_mat).astype(float)
    conf_mat -= q.min(axis=1)[:, None]
    conf_mat /= conf_mat.sum(axis=1)[:, None]
    conf_mat *= errors_to_assign[:, None]

    for i, (row) in enumerate(conf_mat):
        for _ in range(errors_to_assign[i]):
            a = row.argmax()
            result[i, a] += 1
            conf_mat[i, a] -= 1
    return result.astype(np.int)
