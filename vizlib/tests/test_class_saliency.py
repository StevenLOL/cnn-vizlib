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
    def conv_network(self, conv_W):
        il = lasagne.layers.InputLayer((1, 1, 3, 3))
        cl = lasagne.layers.Conv2DLayer(
            il,
            num_filters=1,
            filter_size=(3, 3),
            W=conv_W,
            nonlinearity=lasagne.nonlinearities.identity,
        )
        dl = lasagne.layers.DenseLayer(
            cl, 1, W=np.array([[1]]).astype(theano.config.floatX),
            nonlinearity=lasagne.nonlinearities.sigmoid,
        )
        return dl


    def test_image_occlusion_1_by_1(self):
        # Arrange
        X = np.array([[[[-1, 1, -1],
                        [1, -1, 1],
                        [-1, 1, -1]]]]).astype(theano.config.floatX)
        dl = self.conv_network(conv_W=X[:, :, ::-1, ::-1])

        # Apply
        map = vizlib.class_saliency_map.occlusion(X, dl, 0, square_length=1)

        # Assert
        assert len(set(map.flatten().tolist())) == 1,\
                'All pixels should be equally important'

    def test_image_occlusion_3_by_3(self):
        # Arrange
        X = np.array([[[[-1, 1, -1],
                        [1, -1, 1],
                        [-1, 1, -1]]]]).astype(theano.config.floatX)
        dl = self.conv_network(conv_W=X[:, :, ::-1, ::-1])

        # Apply
        map = vizlib.class_saliency_map.occlusion(X, dl, 0, square_length=3)

        # Assert
        # Pixels in the center more important than at the border, since occluding
        # those will occlude most of the image (which matches perfectly w/ filter)
        # Since the values are identical, sorting order is not guaranteed.
        expected = [4, 1, 3, 5, 7, 0, 2, 6, 8]
        actual = np.lexsort((np.arange(len(map.flatten())), -map.flatten())).tolist()
        assert expected == actual

    def test_image_occlusion_3_by_3_left_part_important(self):
        # Arrange
        X = np.array([[[[-1, 1, -1],
                        [1, -1, 1],
                        [-1, 1, -1]]]]).astype(theano.config.floatX)
        W = np.array([[[[-1, 1, 1],
                        [1, -1, -1],
                        [-1, 1, 1]]]]).astype(theano.config.floatX)
        # Remember, the filter of a convolution is flipped.
        dl = self.conv_network(conv_W=W[:, :, ::-1, ::-1])

        #Apply
        map = vizlib.class_saliency_map.occlusion(X, dl, 0, square_length=3)

        # Assert
        expected = [3, 0, 6, 4, 1, 7, 2, 5, 8]
        actual = np.argsort(-map.flatten()).tolist()
        assert expected == actual
