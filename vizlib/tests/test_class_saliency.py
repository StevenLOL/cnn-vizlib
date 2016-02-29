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

class TestImageOcclusion(object):

    def test_image_occlusion(self):
        # I expect that when I take a convolution layer w/ a filter equal
        # to the input image, and a dense layer that is simply the identity,
        # then the parts of the image that are considered important,
        # should be the same parts that have a large value in the original
        # image.

        # Arrange
        X = np.array([[[[-1, 1, -1],
                        [1, -1, 1],
                        [-1, 1, -1]]]]).astype(theano.config.floatX)
        il = lasagne.layers.InputLayer((1, 1, 3, 3))
        cl = lasagne.layers.Conv2DLayer(
            il,
            num_filters=1,
            filter_size=(3, 3),
            W=X,
        )
        dl = lasagne.layers.DenseLayer(
            cl, 1, W=np.array([[1]]).astype(theano.config.floatX)
        )

        # Apply
        map = vizlib.class_saliency_map.occlusion(X, dl, 0, square_length=1)

        # Assert
        map_order = list(np.argsort(map.flatten()))
        x_order = list(np.argsort(X.flatten()))
        assert map_order == x_order
