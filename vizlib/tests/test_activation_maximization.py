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

        outputs = vizlib.activation_maximization.scores(
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

        outputs = vizlib.activation_maximization.scores(
            X_var, output_layer, ignore_nonlinearity=True)
        assert num_filters == len(outputs)

        # Not much we can test --- just make sure it doesnt crash
        f = theano.function([X_var], outputs, allow_input_downcast=True)
        y = f(X)
        assert num_filters == len(y)

class TestMaximizeScores():

    @classmethod
    def setup_method(self, method):
        np.random.seed(42)

    def simple_conv_network(self, input_shape):
        input_layer = lasagne.layers.InputLayer((None, ) + input_shape)
        conv_layer = lasagne.layers.Conv2DLayer(input_layer, 7, (3, 3))
        dense_layer = lasagne.layers.DenseLayer(conv_layer, 3)
        return dense_layer

    def simple_dataset(self, input_shape):
        X_init = np.random.randn(*((1, ) + input_shape)).astype(theano.config.floatX)
        X_init -= X_init.mean()
        X_init /= X_init.std()
        return X_init

    def scores(self, X, layer):
        return theano.function(
            [],
            vizlib.activation_maximization.scores(X, layer, ignore_nonlinearity=True)
        )()

    def norm(self, X):
        return lasagne.utils.compute_norms(X).max()

    def test_maximize_scores(self):
        # Arrange
        X_init = self.simple_dataset((3, 32, 32))
        network = self.simple_conv_network((3, 32, 32))
        X = theano.shared(X_init)
        initial_norm = lasagne.utils.compute_norms(X_init).max()
        initial_scores = self.scores(X, network)

        # Apply
        scores_and_maximizers = vizlib.activation_maximization.maximize_scores(
            network,
            X_init,
            number_of_iterations=100,
            max_norm=initial_norm
        )
        scores, maximizers = zip(*scores_and_maximizers)

        # Assert
        np.testing.assert_array_less(initial_scores, scores)
        for x in maximizers:
            assert np.isclose(self.norm(X.get_value()), initial_norm)
