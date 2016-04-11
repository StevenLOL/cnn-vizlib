#!/usr/bin/env python2.7
# encoding: utf-8
'''
'''
from netshared import *

def parse_args(args):
    if len(args) != 1:
        raise ValueError('Expected you to pass in "ndense" as extra parameter')
    ndense = int(args[0])
    return ndense

def build(ds, nepochs, lr, momentum, args):
    ndense = parse_args(args)

    layers=[
        ('i', lasagne.layers.InputLayer),
    ]
    params = {}
    for i in range(1, ndense+1):
        layers.append(('c{i}'.format(**locals()), Conv2DLayer))
        layers.append(('p{i}'.format(**locals()), MaxPool2DLayer))
        params['c{i}_num_filters'.format(**locals())] = 3
        params['c{i}_filter_size'.format(**locals())] = (3, 3)
        params['c{i}_pad'.format(**locals())] = 'same'
        params['p{i}_pool_size'.format(**locals())] = (2, 2)

    layers.append(('o', lasagne.layers.DenseLayer))

    net = NeuralNet(
        layers=[
            ('i', lasagne.layers.InputLayer),
            ('d', lasagne.layers.DenseLayer),
            ('o', lasagne.layers.DenseLayer),
        ],
        i_shape=(None, ) + ds.X.shape[1:],

        d_num_units=ndense,
        d_nonlinearity=nonlinearities.sigmoid,

        o_num_units=len(set(ds.y)),
        o_nonlinearity=nonlinearities.softmax,

        update_learning_rate=theano.shared(np.float32(lr[0])),
        update_momentum=theano.shared(np.float32(momentum[0])),

        regression=False,
        on_epoch_finished=[
            AdjustVariable('update_learning_rate', start=lr[0], stop=lr[1]),
            AdjustVariable('update_momentum', start=momentum[0], stop=momentum[1]),
            EarlyStopping(patience=200),
        ],
        max_epochs=nepochs,
        verbose=1,

        **params
    )
    return net

def name(args):
    ndense = parse_args(args)
    return 'd{ndense}'.format(**locals())

