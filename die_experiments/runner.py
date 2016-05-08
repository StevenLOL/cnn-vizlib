#!/usr/bin/env python2.7
# encoding: utf-8
"""
"""
import importlib
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
import os
import errno

try:
    from lasagne.layers.cuda_convnet import Conv2DCCLayer as Conv2DLayer
    from lasagne.layers.cuda_convnet import MaxPool2DCCLayer as MaxPool2DLayer
except ImportError:
    Conv2DLayer = layers.Conv2DLayer
    MaxPool2DLayer = layers.MaxPool2DLayer


sys.setrecursionlimit(10000)  # for pickle...
np.random.seed(42)

def ensure_dir(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

def load_die_dataset():
    X = np.load('../die/die.32.X.npy')[:, None, :, :].astype(np.float32)
    y = np.load('../die/die.y.npy').astype(np.int32) - 1
    ds = vizlib.data.DataSet(X, y).standardize('global')
    return ds.shuffle()

def plot_weights(weights):
    fig = pyplot.figure(figsize=(6, 6))
    fig.subplots_adjust(
        left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)

    for i in range(16):
        ax = fig.add_subplot(4, 4, i + 1, xticks=[], yticks=[])
        ax.imshow(weights[:, i].reshape(96, 96), cmap='gray')
    pyplot.show()


def save_net(savedir, net):
    ensure_dir(savedir)
    now = datetime.datetime.utcnow()
    nowfmt = now.isoformat()
    with open(os.path.join(savedir, '{}_net.pickle'.format(nowfmt)), 'wb') as f:
        pickle.dump(net, f, -1)

if __name__ == '__main__':
    netfname = os.path.splitext(sys.argv[1])[0]
    netmodule = importlib.import_module(netfname)
    netname = netmodule.name(sys.argv[2:])
    print('saving to', netname)

    ds = load_die_dataset()
    lr = (1e-4, 0.0001)
    momentum = (0.9, 0.999)
    nepochs = 10000

    for i in range(5):
        net = netmodule.build(ds, nepochs, lr, momentum, sys.argv[2:])
        net.fit(ds.X, ds.y)
        save_net(netname, net)
