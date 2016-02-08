import vizlib
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

        outputs = vizlib.activation_maximization.maximize.expression_to_maximize(
            X_var, output_layer, ignore_nonlinearity=True)
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

        outputs = vizlib.activation_maximization.maximize.expression_to_maximize(
            X_var, output_layer, ignore_nonlinearity=True)
        assert num_filters == len(outputs)

        # Not much we can test --- just make sure it doesnt crash
        f = theano.function([X_var], outputs, allow_input_downcast=True)
        y = f(X)
        assert num_filters == len(y)

class TestMaximizeScores():

    def test_maximize_scores(self):

        np.random.seed(42)
        X_init = np.random.randn(1, 3, 32, 32).astype(theano.config.floatX)
        X = theano.shared(X_init)
        input_layer = lasagne.layers.InputLayer((None, ) + X_init.shape[1:])
        conv_layer = lasagne.layers.Conv2DLayer(input_layer, 7, (3, 3))
        dense_layer = lasagne.layers.DenseLayer(conv_layer, 3)
        n_iterations = 100

        # Test our function by making sure that the activation expression
        # has a lower score initially than it has at the end
        expressions = vizlib.activation_maximization.maximize.expression_to_maximize(
            X, dense_layer, ignore_nonlinearity=True)

        f = theano.function([], expressions)
        initial_scores = f()

        scores_and_maximizers = vizlib.activation_maximization.maximize_score(
            dense_layer, X_init, n_iterations)

        np.testing.assert_array_less(
            initial_scores,
            -np.array([s for s, _ in scores_and_maximizers]),
        )
