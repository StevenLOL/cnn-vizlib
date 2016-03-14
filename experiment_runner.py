'''Runs the experiment defined by a network and a set of experiment parameters.

The settings file that is expected as an argument is a simple python file,
that defines the following:

    from lasagne.layers import *
    from lasagne.nonlinearities import *

    dataset = vizlib.data.die()
    serialization_folder = 'results'

    # A set of learning rate parameters.
    # If a list of values is given for a key, each of these values in the list
    # is combined w/ all other values for other keys.
    params = {
        'learning_rate': 1e-2,
        'momentum': 0.9,
        'l1': [0, 1e-1, 1e-2],
        'l2': 0,
        'max_epochs': 100,
    }

    # A list of network configurations.
    # Each entry in the list represents a different network configuration.
    # Each entry in the configuration represents a set of parameters.
    # Based on the parameter names, a different layer is instantiated.
    # Configuration below represents two networks, one w/ a conv layer
    # and one with a sigmoidal dense layer.
    network = [
        [
            {'num_filters': [...], 'filter_size': (3, 3)},
        ],
        [
            {'num_units': [10, 20, 30]}
        ],
    ]

Usage:
    experiment_runner <settings>

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
from lasagne.layers import *
from lasagne.nonlinearities import *

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

def experiment_name(base_name, params, network):
    return '{}_{}_{}'.format(base_name, describe_params(params), describe_network(network))

def describe_params(params):
    # Let's be super lazy!!!
    parts = ['{}={}'.format(k, params[k]) for k in sorted(params.keys())]
    return '.'.join(parts)

def describe_network(network):
    return '.'.join([describe_params(layer) for layer in network])

def create_serialize_network(serialization_folder, experiment_name):
    closure = dict(epoch_count=0)

    def serialize_network(nn, history):
        output_layer = nn.layers_[-1]

        fname = os.path.join(serialization_folder, '{epoch_count}_{experiment_name}.p'.format(
            epoch_count=closure['epoch_count'],
            experiment_name=experiment_name,
        ))

        with open(fname, 'wb') as fh:
            pickle.dump(output_layer, fh, -1)
        closure['epoch_count'] = closure['epoch_count'] + 1

    return serialize_network

def create_serialize_history(serialization_folder, experiment_name):
    parameterized_name = os.path.join(serialization_folder, '{}_history.p'.format(experiment_name))
    plot_name = os.path.join(serialization_folder, '{}_history.png'.format(experiment_name))

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

def ensure_list(params):
    for k in params:
        v = params[k]
        if not isinstance(v, list):
            params[k] = [v]

def possible_parameter_settings(params):
    ensure_list(params)
    for opt in itertools.product(*params.values()):
        yield dict(zip(params.keys(), opt))

def possible_network_settings(meta_network_setting, dataset):
    settings = []
    for meta_layer_setting in meta_network_setting:
        # layer settings are themselves nested, of which we need all combinations.
        ensure_list(meta_layer_setting)
        settings.append(list(possible_parameter_settings(meta_layer_setting)))

    for network_setting in itertools.product(*settings):
        yield network_setting

def build_network(network_settings):
    il = InputLayer((None, ) + dataset.shape[1:])
    for l in network_settings:
        il = construct_layer(il, l)
    ol = DenseLayer(il, num_units=len(dataset.classes), nonlinearity=softmax)
    return ol

def construct_layer(input, l):
    if 'num_units' in l:
        return DenseLayer(input, num_units=l['num_units'], nonlinearity=l.get('nonlinearity', sigmoid))
    if 'num_filters' in l:
        return Conv2DLayer(
            input,
            num_filters=l['num_filters'],
            filter_size=l.get('filter_size', (3, 3)),
            pad=l.get('pad', 'same'),
        )
    if 'pool_size' in l:
        return Pool2DLayer(
            input,
            pool_size=l['pool_size'],
        )
    raise ValueError('Could not match the configuration to a specific layer: {}'.format(l.keys()))

if __name__ == '__main__':
    args = docopt.docopt(__doc__)
    settings_file = args['<settings>']
    if settings_file.endswith('.py'):
        settings_file = settings_file[:-len('.py')]
    settings = importlib.import_module(settings_file)

    meta_params = settings.params
    dataset = settings.dataset
    serialization_folder = settings.serialization_folder
    meta_network_settings = settings.network

    for meta_network_setting in meta_network_settings:
        for network_settings in possible_network_settings(meta_network_setting, dataset):
            nw = build_network(network_settings)
            pickled_network = pickle.dumps(nw, -1)

            for params in possible_parameter_settings(meta_params):
                # Load the pickled nw to have the exact same intialization
                nw = pickle.loads(pickled_network)
                exp_name = experiment_name(settings_file, params, network_settings)
                print(exp_name)
                calc_confusion_matrix, update_confusion_matrix_history, add_confusion_matrixes_to_history =\
                        create_confusion_matrix_functions()
                serialize_network_callback = create_serialize_network(serialization_folder, exp_name)
                serialize_history_callback = create_serialize_history(serialization_folder, exp_name)

                nn = nolearn.lasagne.NeuralNet(
                    nw,

                    update_learning_rate=params['learning_rate'],
                    update_momentum=params['momentum'],

                    objective_l1=params['l1'],
                    objective_l2=params['l2'],

                    max_epochs=params['max_epochs'],

                    custom_score=('confusion_matrix', calc_confusion_matrix),
                    on_epoch_finished=[update_confusion_matrix_history, serialize_network_callback],
                    on_training_finished=[add_confusion_matrixes_to_history, serialize_history_callback],

                    verbose=0
                )

                nn.fit(dataset.X, dataset.y)
