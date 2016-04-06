#!/usr/bin/env python2.7
# encoding: utf-8
"""
"""

import cPickle as pickle
import sys
import datetime

from matplotlib import pyplot
import numpy as np
from lasagne import layers
from lasagne import nonlinearities
from nolearn.lasagne import NeuralNet
import theano
import vizlib

try:
    from lasagne.layers.cuda_convnet import Conv2DCCLayer as Conv2DLayer
    from lasagne.layers.cuda_convnet import MaxPool2DCCLayer as MaxPool2DLayer
except ImportError:
    Conv2DLayer = layers.Conv2DLayer
    MaxPool2DLayer = layers.MaxPool2DLayer


sys.setrecursionlimit(10000)  # for pickle...
np.random.seed(42)

def load_die_dataset():
    X = np.load('die/die.32.X.npy')[:, None, :, :].astype(np.float32)
    y = np.load('die/die.y.npy').astype(np.int32) - 1
    ds = vizlib.data.DataSet(X, y).standardize('global')
    return ds

def plot_weights(weights):
    fig = pyplot.figure(figsize=(6, 6))
    fig.subplots_adjust(
        left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)

    for i in range(16):
        ax = fig.add_subplot(4, 4, i + 1, xticks=[], yticks=[])
        ax.imshow(weights[:, i].reshape(96, 96), cmap='gray')
    pyplot.show()


class AdjustVariable(object):
    def __init__(self, name, start=0.03, stop=0.001):
        self.name = name
        self.start, self.stop = start, stop
        self.ls = None

    def __call__(self, nn, train_history):
        if self.ls is None:
            self.ls = np.linspace(self.start, self.stop, nn.max_epochs)

        epoch = train_history[-1]['epoch']
        new_value = np.cast['float32'](self.ls[epoch - 1])
        getattr(nn, self.name).set_value(new_value)


class EarlyStopping(object):
    def __init__(self, patience=100):
        self.patience = patience
        self.best_valid = np.inf
        self.best_valid_epoch = 0
        self.best_weights = None

    def __call__(self, nn, train_history):
        current_valid = train_history[-1]['valid_loss']
        current_epoch = train_history[-1]['epoch']
        if current_valid < self.best_valid:
            self.best_valid = current_valid
            self.best_valid_epoch = current_epoch
            self.best_weights = nn.get_all_params_values()
        elif self.best_valid_epoch + self.patience < current_epoch:
            print("Early stopping.")
            print("Best valid loss was {:.6f} at epoch {}.".format(
                self.best_valid, self.best_valid_epoch))
            nn.load_params_from(self.best_weights)
            raise StopIteration()

def save_net(net):
    now = datetime.datetime.utcnow()
    nowfmt = now.isoformat()
    with open('{}_net.pickle'.format(nowfmt), 'wb') as f:
        pickle.dump(net, f, -1)

if __name__ == '__main__':
    ds = load_die_dataset()

    lr = 1e-4
    momentum = 0.9

    net = NeuralNet(
        layers=[
            ('i', layers.InputLayer),
            ('c1', Conv2DLayer),
            ('p1', MaxPool2DLayer),
            ('o', layers.DenseLayer),
        ],
        i_shape=(None, ) + ds.X.shape[1:],

        c1_num_filters=3, c1_filter_size=(3, 3), p1_pool_size=(2, 2),

        o_num_units=len(set(ds.y)), o_nonlinearity=nonlinearities.softmax,

        update_learning_rate=theano.shared(np.float32(lr)),
        update_momentum=theano.shared(np.float32(momentum)),

        regression=False,
        on_epoch_finished=[
            AdjustVariable('update_learning_rate', start=lr, stop=0.0001),
            AdjustVariable('update_momentum', start=momentum, stop=0.999),
            EarlyStopping(patience=200),
        ],
        max_epochs=3000,
        verbose=1,
    )

    net.fit(ds.X, ds.y)
    save_net(net)
