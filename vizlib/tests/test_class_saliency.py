'''
'''
import lasagne
import theano
import theano.tensor as T
import vizlib
import numpy as np


def test_taylor_series():
    X_var = T.tensor4('X_var')
    num_units = 2
    input_layer = lasagne.layers.InputLayer((None, 3, 32, 32), X_var)
    output_layer = lasagne.layers.DenseLayer(input_layer, num_units)

    X = np.random.randn(1, 3, 32, 32).astype(theano.config.floatX)
    fs = vizlib.class_saliency_map.taylor_expansion_functions(output_layer)

    assert len(fs) == num_units
    for f in fs:
        assert f(X).shape == (32, 32)

def test_image_occlusion():
    X_var = T.tensor4('X_var')
    num_units = 2
    input_layer = lasagne.layers.InputLayer((None, 3, 32, 32), X_var)
    output_layer = lasagne.layers.DenseLayer(input_layer, num_units)

    X = np.random.randn(1, 3, 32, 32).astype(theano.config.floatX)

    m = vizlib.class_saliency_map.occlusion(X, output_layer, 0)

    assert m.shape == (32, 32)
