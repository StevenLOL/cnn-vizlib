import vizlib.activation_maximization as am
import lasagne
import lasagne.layers
import theano
import theano.tensor as T
import numpy as np


class TestExpressionToMaximize():

    def test_dense_output_layer(self):

        X = np.zeros((1, 3, 32, 32))
        X_var = T.tensor4('X_var')
        input_layer = lasagne.layers.InputLayer((None, ) + X.shape[1:], X_var)
        num_units = 5
        output_layer = lasagne.layers.DenseLayer(input_layer, num_units=num_units)

        outputs = am.expression_to_maximize(output_layer, ignore_nonlinearity=True)
        assert num_units == len(outputs)

        # Not much we can test --- just make sure it doesnt crash
        f = theano.function([X_var], outputs, allow_input_downcast=True)
        y = f(X)
        assert num_units == len(y)

    def test_conv_output_layer(self):

        X = np.zeros((1, 3, 32, 32))
        X_var = T.tensor4('X_var')
        input_layer = lasagne.layers.InputLayer((None, ) + X.shape[1:], X_var)
        num_filters = 12
        output_layer = lasagne.layers.Conv2DLayer(input_layer, num_filters, (3, 3))

        outputs = am.expression_to_maximize(output_layer, ignore_nonlinearity=True)
        assert num_filters == len(outputs)

        # Not much we can test --- just make sure it doesnt crash
        f = theano.function([X_var], outputs, allow_input_downcast=True)
        y = f(X)
        assert num_filters == len(y)
