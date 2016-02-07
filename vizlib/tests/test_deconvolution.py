from vizlib import deconvolution
import numpy as np
import theano
import theano.tensor as T
import lasagne.layers

class TestMaxSwitch2DLayer():

    def test_something(self):

        X = np.array([[1, 2, 3, 4],
                      [5, 6, 5, 1],
                      [9, 0, 1, 9],
                      [3, 4, 5, 6]])
        X = X.reshape((1, 1, ) + X.shape)
        expected_switches = np.array([[3, 2],
                                      [0, 1]])
        expected_switches = expected_switches.reshape(
            (1, 1) + expected_switches.shape)

        X_var = T.tensor4('X')
        input_layer = lasagne.layers.InputLayer(X.shape, X_var)
        layer = deconvolution.MaxSwitch2DLayer(input_layer, (2, 2))
        #layer = lasagne.layers.MaxPool2DLayer(input_layer, (2, 2))

        expr = layer.get_output_for(X_var)
        f = theano.function([X_var], expr, allow_input_downcast=True)

        switches = f(X)
        np.testing.assert_array_equal(expected_switches, switches)

def test_unpool():

    X = np.array([[
        [[6, 5],
         [9, 9]],
        [[8, 7],
         [3, 2]],
        [[1, 6],
         [4, 2]]
    ]])
    switches = np.array([[
        [[3, 2],
         [0, 1]],
        [[1, 2],
         [3, 3]],
        [[3, 2],
         [1, 0]],
    ]])
    expected_output = np.array([[
        [[0, 0, 0, 0],
         [0, 6, 5, 0],
         [9, 0, 0, 9],
         [0, 0, 0, 0]],
        [[0, 8, 0, 0],
         [0, 0, 7, 0],
         [0, 0, 0, 0],
         [0, 3, 0, 2]],
        [[0, 0, 0, 0],
         [0, 1, 6, 0],
         [0, 4, 2, 0],
         [0, 0, 0, 0]],
    ]])

    output = deconvolution.unpool(X, switches,
                                  pool_size=(2, 2), 
                                  stride=(2, 2),
                                  padding=(0, 0))
    np.testing.assert_array_equal(expected_output, output)
