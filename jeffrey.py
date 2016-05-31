#!/usr/bin/env python2.7
# encoding: utf-8
'''
'''
from __future__ import print_function
# Load jeffreys model and try to make some predictions
import sys
sys.path.append('../kaggle_diabetic_retinopathy/')
import cPickle as pickle
import re
import glob
import os

import time

import lasagne as nn
from lasagne.layers import dnn
from lasagne.nonlinearities import LeakyRectify

import theano
import theano.tensor as T
import numpy as np
import pandas as pd
from generators import DataLoader
from layers import ApplyNonlinearity

from utils import hms, architecture_string, get_img_ids_from_iter

def from_raw():
    layers = []
    batch_size = 2

    num_channels = 3
    input_height = input_width = 512
    leakiness = 0.5

    l_in_imgdim = nn.layers.InputLayer(
        shape=(batch_size, 2),
        name='imgdim'
    )

    l_in1 = nn.layers.InputLayer(
        shape=(batch_size, num_channels, input_width, input_height),
        name='images'
    )
    layers.append(l_in1)

    Conv2DLayer = dnn.Conv2DDNNLayer
    MaxPool2DLayer = dnn.MaxPool2DDNNLayer
    DenseLayer = nn.layers.DenseLayer

    l_conv = Conv2DLayer(layers[-1],
                         num_filters=32, filter_size=(7, 7), stride=(2, 2),
                         pad='same',
                         nonlinearity=LeakyRectify(leakiness),
                         W=nn.init.Orthogonal(1.0), b=nn.init.Constant(0.1),
                         untie_biases=True)
    layers.append(l_conv)

    l_pool = MaxPool2DLayer(layers[-1], pool_size=(3, 3), stride=(2, 2))
    layers.append(l_pool)

    l_conv = Conv2DLayer(layers[-1],
                         num_filters=32, filter_size=(3, 3), stride=(1, 1),
                         pad='same',
                         nonlinearity=LeakyRectify(leakiness),
                         W=nn.init.Orthogonal(1.0), b=nn.init.Constant(0.1),
                         untie_biases=True)
    layers.append(l_conv)

    l_conv = Conv2DLayer(layers[-1],
                         num_filters=32, filter_size=(3, 3), stride=(1, 1),
                         pad='same',
                         nonlinearity=LeakyRectify(leakiness),
                         W=nn.init.Orthogonal(1.0), b=nn.init.Constant(0.1),
                         untie_biases=True)
    layers.append(l_conv)

    l_pool = MaxPool2DLayer(layers[-1], pool_size=(3, 3), stride=(2, 2))
    layers.append(l_pool)

    l_conv = Conv2DLayer(layers[-1],
                         num_filters=64, filter_size=(3, 3), stride=(1, 1),
                         pad='same',
                         nonlinearity=LeakyRectify(leakiness),
                         W=nn.init.Orthogonal(1.0), b=nn.init.Constant(0.1),
                         untie_biases=True)
    layers.append(l_conv)

    l_conv = Conv2DLayer(layers[-1],
                         num_filters=64, filter_size=(3, 3), stride=(1, 1),
                         pad='same',
                         nonlinearity=LeakyRectify(leakiness),
                         W=nn.init.Orthogonal(1.0), b=nn.init.Constant(0.1),
                         untie_biases=True)
    layers.append(l_conv)

    l_pool = MaxPool2DLayer(layers[-1], pool_size=(3, 3), stride=(2, 2))
    layers.append(l_pool)

    l_conv = Conv2DLayer(layers[-1],
                         num_filters=128, filter_size=(3, 3), stride=(1, 1),
                         pad='same',
                         nonlinearity=LeakyRectify(leakiness),
                         W=nn.init.Orthogonal(1.0), b=nn.init.Constant(0.1),
                         untie_biases=True)
    layers.append(l_conv)

    l_conv = Conv2DLayer(layers[-1],
                         num_filters=128, filter_size=(3, 3), stride=(1, 1),
                         pad='same',
                         nonlinearity=LeakyRectify(leakiness),
                         W=nn.init.Orthogonal(1.0), b=nn.init.Constant(0.1),
                         untie_biases=True)
    layers.append(l_conv)

    l_conv = Conv2DLayer(layers[-1],
                         num_filters=128, filter_size=(3, 3), stride=(1, 1),
                         pad='same',
                         nonlinearity=LeakyRectify(leakiness),
                         W=nn.init.Orthogonal(1.0), b=nn.init.Constant(0.1),
                         untie_biases=True)
    layers.append(l_conv)

    l_conv = Conv2DLayer(layers[-1],
                         num_filters=128, filter_size=(3, 3), stride=(1, 1),
                         pad='same',
                         nonlinearity=LeakyRectify(leakiness),
                         W=nn.init.Orthogonal(1.0), b=nn.init.Constant(0.1),
                         untie_biases=True)
    layers.append(l_conv)

    l_pool = MaxPool2DLayer(layers[-1], pool_size=(3, 3), stride=(2, 2))
    layers.append(l_pool)

    l_conv = Conv2DLayer(layers[-1],
                         num_filters=256, filter_size=(3, 3), stride=(1, 1),
                         pad='same',
                         nonlinearity=LeakyRectify(leakiness),
                         W=nn.init.Orthogonal(1.0), b=nn.init.Constant(0.1),
                         untie_biases=True)
    layers.append(l_conv)

    l_conv = Conv2DLayer(layers[-1],
                         num_filters=256, filter_size=(3, 3), stride=(1, 1),
                         pad='same',
                         nonlinearity=LeakyRectify(leakiness),
                         W=nn.init.Orthogonal(1.0), b=nn.init.Constant(0.1),
                         untie_biases=True)
    layers.append(l_conv)

    l_conv = Conv2DLayer(layers[-1],
                         num_filters=256, filter_size=(3, 3), stride=(1, 1),
                         pad='same',
                         nonlinearity=LeakyRectify(leakiness),
                         W=nn.init.Orthogonal(1.0), b=nn.init.Constant(0.1),
                         untie_biases=True)
    layers.append(l_conv)

    l_conv = Conv2DLayer(layers[-1],
                         num_filters=256, filter_size=(3, 3), stride=(1, 1),
                         pad='same',
                         nonlinearity=LeakyRectify(leakiness),
                         W=nn.init.Orthogonal(1.0), b=nn.init.Constant(0.1),
                         untie_biases=True)
    layers.append(l_conv)
    l_pool = MaxPool2DLayer(layers[-1], pool_size=(3, 3), stride=(2, 2),
                            name='coarse_last_pool')
    layers.append(l_pool)

    layers.append(nn.layers.DropoutLayer(layers[-1], p=0.5))
    layers.append(DenseLayer(layers[-1],
                             nonlinearity=None,
                             num_units=1024,
                             W=nn.init.Orthogonal(1.0),
                             b=nn.init.Constant(0.1),
                             name='first_fc_0'))
    l_pool = nn.layers.FeaturePoolLayer(layers[-1],
                                        pool_size=2,
                                        pool_function=T.max)
    layers.append(l_pool)

    l_first_repr = layers[-1]

    l_coarse_repr = nn.layers.concat([l_first_repr,
                                      l_in_imgdim])
    layers.append(l_coarse_repr)

    # Combine representations of both eyes.
    layers.append(
        nn.layers.ReshapeLayer(layers[-1], shape=(batch_size // 2, -1)))

    layers.append(nn.layers.DropoutLayer(layers[-1], p=0.5))
    layers.append(nn.layers.DenseLayer(layers[-1],
                                       nonlinearity=None,
                                       num_units=1024,
                                       W=nn.init.Orthogonal(1.0),
                                       b=nn.init.Constant(0.1),
                                       name='combine_repr_fc'))
    l_pool = nn.layers.FeaturePoolLayer(layers[-1],
                                        pool_size=2,
                                        pool_function=T.max)
    layers.append(l_pool)

    l_hidden = nn.layers.DenseLayer(nn.layers.DropoutLayer(layers[-1], p=0.5),
                                    num_units=output_dim * 2,
                                    nonlinearity=None,  # No softmax yet!
                                    W=nn.init.Orthogonal(1.0),
                                    b=nn.init.Constant(0.1))
    layers.append(l_hidden)

    # Reshape back to 5.
    layers.append(nn.layers.ReshapeLayer(layers[-1],
                                         shape=(batch_size, 5)))

    # Apply softmax.
    l_out = ApplyNonlinearity(layers[-1],
                              nonlinearity=nn.nonlinearities.softmax)
    layers.append(l_out)

    l_ins = [l_in1, l_in_imgdim]

    nn.layers.set_all_param_values(l_out, pickle.load(open(RAW_DUMP_PATH, 'rb')))

    return l_out, l_ins

DUMP_PATH = '../kaggle_diabetic_retinopathy/dumps/2015_07_17_123003.pkl'
RAW_DUMP_PATH = '../kaggle_diabetic_retinopathy/dumps/2015_07_17_123003_PARAMSDUMP.pkl'
NEW_DUMP_PATH = '../kaggle_diabetic_retinopathy/dumps/2015_07_17_123003.updated.pkl'
IMG_DIR = '../train_ds2_crop/'

model_data = pickle.load(open(DUMP_PATH, 'r'))

train_labels = pd.read_csv(os.path.join('../kaggle_diabetic_retinopathy/data/trainLabels.csv'))
batch_size = model_data['batch_size']
output_dim = 5
chunk_size = model_data['chunk_size']
patient_ids = sorted(set(get_img_ids_from_iter(train_labels.image)))
no_transfo_params = model_data['data_loader_params']['no_transfo_params']
if 'paired_transfos' in model_data:
    paired_transfos = model_data['paired_transfos']
else:
    paired_transfos = False

# Overwrite the model w/ the updated model (one that is suitable for use w/ newer versions of lasagne).
l_out, l_ins = from_raw()
model_data['l_out'] = l_out
model_data['l_ins'] = l_ins
with open(NEW_DUMP_PATH, 'wb') as fh:
    pickle.dump(model_data, fh, protocol=-1)
print('Dumped new model')

output = nn.layers.get_output(l_out, deterministic=True)
input_ndims = [len(nn.layers.get_output_shape(l_in))
               for l_in in l_ins]
xs_shared = [nn.utils.shared_empty(dim=ndim)
             for ndim in input_ndims]
idx = T.lscalar('idx')

givens = {}
for l_in, x_shared in zip(l_ins, xs_shared):
    givens[l_in.input_var] = x_shared[idx * batch_size:(idx + 1) * batch_size]

compute_output = theano.function(
    [idx],
    output,
    givens=givens,
    on_unused_input='ignore'
)
data_loader = DataLoader()
new_dataloader_params = model_data['data_loader_params']
new_dataloader_params.update({'images_test': patient_ids})
new_dataloader_params.update({'labels_test': train_labels.level.values})
new_dataloader_params.update({'prefix_train': IMG_DIR})
data_loader.set_params(new_dataloader_params)
num_chunks = int(np.ceil((2 * len(patient_ids)) / float(chunk_size)))

def do_pred(img_ids):
    test_gen = lambda: data_loader.create_fixed_gen(
        img_ids,
        chunk_size=chunk_size,
        prefix_train=IMG_DIR,
        prefix_test=IMG_DIR,
        transfo_params=no_transfo_params,
        paired_transfos=paired_transfos,
    )
    outputs = []

    for e, (xs_chunk, chunk_shape, chunk_length) in enumerate(test_gen()):
        num_batches_chunk = int(np.ceil(chunk_length / float(batch_size)))

        print("Chunk %i/%i" % (e + 1, num_chunks))

        print("  load data onto GPU")
        for x_shared, x_chunk in zip(xs_shared, xs_chunk):
            x_shared.set_value(x_chunk)

        print("  compute output in batches")
        outputs_chunk = []
        for b in xrange(num_batches_chunk):
            out = compute_output(b)
            outputs_chunk.append(out)

        outputs_chunk = np.vstack(outputs_chunk)
        outputs_chunk = outputs_chunk[:chunk_length]

        outputs.append(outputs_chunk)

    return np.vstack(outputs), xs_chunk

