#!/usr/bin/env python2.7
# encoding: utf-8
'''Runs the experiment defined by a network and a set of experiment parameters.

The settings file that is expected as an argument is a simple python file,
that defines the following:

    from lasagne.layers import *
    from lasagne.nonlinearities import *

    dataset = vizlib.data.die()

    params = {
        'serialization_folder': '',
        'experiment_name': 'myname',
        'learning_rate': 1e-2,
        'momentum': 0.9,
        'l1': 0,
        'l2': 0,
        'max_epochs': 100,
    }

    # A network, where the network = ... is the final output layer.
    il = InputLayer((None, ) + ds.shape[1:])
    ...
    network = DenseLayer(
        il,
        num_units=len(dataset.classes),
        nonlinearity=softmax
    )

Usage:
    test_runner <settings>

'''
from __future__ import print_function
import nolearn.lasagne
import numpy as np
import docopt
import importlib
import os
from sklearn.metrics import confusion_matrix
import pickle
import matplotlib.pyplot as plt
import itertools

listable_params = ('learning_rate', 'momentum', 'l1', 'l2')

def plot_history(net):
    train_history = net.train_history_
    train_loss = np.array([row['train_loss'] for row in train_history])
    valid_loss = np.array([row['valid_loss'] for row in train_history])
    valid_acc = [row['valid_accuracy'] for row in train_history]
    train_acc = [row['train_accuracy'] for row in train_history]

    fig, ax1 = plt.subplots()

    ax2 = ax1.twinx()
    train_acc_handle, = ax2.plot(train_acc, 'b.--')
    valid_acc_handle, = ax2.plot(valid_acc, 'r.--')
    ax2.set_ylabel('accuracy')

    train_loss_handle, = ax1.plot(-train_loss, 'b-')
    valid_loss_handle, = ax1.plot(-valid_loss, 'r-')
    ax1.set_xlabel('epochs')
    # Make the y-axis label and tick labels match the line color.
    ax1.set_ylabel('$-$cross-entropy')

    plt.legend(
        [train_acc_handle, valid_acc_handle, train_loss_handle, valid_loss_handle],
        ['train accuracy', 'validation accuracy', 'train cross-entroy', 'validation cross-entropy'],
        loc='lower center'
    )
    return fig

def create_serialize_network(params):
    closure = dict(epoch_count=0)

    # Need to serialize the network to a file w/ the name:
    parameterized_name = os.path.join(params['serialization_folder'], '{experiment_name}_{learning_rate}_{momentum}_{l1}_{l2}_{epoch_count}.p')
    def serialize_network(nn, history):
        output_layer = nn.layers_[-1]

        fmt_params = dict(epoch_count=closure['epoch_count'])
        fmt_params.update(params)
        fname = parameterized_name.format(**fmt_params)

        with open(fname, 'wb') as fh:
            pickle.dump(output_layer, fh, -1)
        closure['epoch_count'] = closure['epoch_count'] + 1

    return serialize_network

def create_serialize_history(params):
    parameterized_name = os.path.join(params['serialization_folder'], '{experiment_name}_{learning_rate}_{momentum}_{l1}_{l2}_history.p')
    # Also create a plot --- just for convenience
    plot_name = os.path.join(params['serialization_folder'], '{experiment_name}_{learning_rate}_{momentum}_{l1}_{l2}_history.png')

    def serialize_history(nn, history):
        fname = parameterized_name.format(**params)
        with open(fname, 'wb') as fh:
            pickle.dump(history, fh, -1)
        fig = plot_history(nn)
        fig.savefig(plot_name.format(**params))
        print('Saved to:', plot_name.format(**params))
        print('Best accuracy:', np.max([r['valid_accuracy'] for r in history]))
        plt.close(fig)

    return serialize_history

def create_confusion_matrix_functions():
    '''nolearn.lasagne does not allow access to the y_true, y_pred
    in the `on_batch_finished` and `on_epoch_finished` hooks.

    only the `custom_score` hook, which is only run for the `validation`
    set has access to this information.

    unfortunately this information is aggregated using `np.mean`, meaning
    the score can only be a single scalar.

    this is why we record history ourselves here, and then rewrite history
    in the `on_training_finished` hook.
    '''
    closure = dict(
        batch_history=[],
        confmat_history=[],
    )
    def calc_confusion_matrix(yb, y_prob):
        y_pred = y_prob.argmax(axis=1)
        closure['batch_history'].append(confusion_matrix(yb, y_pred))
        return 0

    def update_confusion_matrix_history(nn, train_history):
        conf_mat = np.sum(closure['batch_history'], axis=0)
        closure['confmat_history'].append(conf_mat)
        closure['batch_history'] = []

    def add_confusion_matrixes_to_history(nn, train_history):
        for h, info in zip(closure['confmat_history'], train_history):
            info['confusion_matrix'] = h

    return (
        calc_confusion_matrix,
        update_confusion_matrix_history,
        add_confusion_matrixes_to_history
    )

def ensure_list(params, k):
    v = params[k]
    if not isinstance(v, list):
        params[k] = [v]

def reduce_params(params):
    opts = [params[k] for k in listable_params]
    for opt in itertools.product(*opts):
        param_copy = {}
        param_copy.update(params)
        for k, v in zip(listable_params, opt):
            param_copy[k] = v
        yield param_copy

if __name__ == '__main__':
    args = docopt.docopt(__doc__)
    settings_file = args['<settings>']
    if settings_file.endswith('.py'):
        settings_file = settings_file[:-len('.py')]
    settings = importlib.import_module(settings_file)

    params = settings.params
    dataset = settings.dataset
    network = settings.network
    pickled_network = pickle.dumps(network, -1)

    for v in listable_params:
        ensure_list(params, v)

    for params in reduce_params(params):
        network = pickle.loads(pickled_network)
        print(params)
        calc_confusion_matrix, update_confusion_matrix_history, add_confusion_matrixes_to_history =\
                create_confusion_matrix_functions()
        serialize_network = create_serialize_network(params)
        serialize_history = create_serialize_history(params)

        nn = nolearn.lasagne.NeuralNet(
            network,

            update_learning_rate=params['learning_rate'],
            update_momentum=params['momentum'],

            objective_l1=params['l1'],
            objective_l2=params['l2'],

            max_epochs=params['max_epochs'],

            custom_score=('confusion_matrix', calc_confusion_matrix),
            on_epoch_finished=[update_confusion_matrix_history, serialize_network],
            on_training_finished=[add_confusion_matrixes_to_history, serialize_history],

            verbose=0
        )

        nn.fit(dataset.X, dataset.y)
