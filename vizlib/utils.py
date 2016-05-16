'''
'''
import nolearn.lasagne
import lasagne
import theano
import theano.tensor as T
import numpy as np
from time import time


def get_input_var(output_layer):
    layer = output_layer
    while not hasattr(layer, 'input_var'):
        layer = layer.input_layer
    return layer.input_var

def get_input_vars(output_layer):
    layer = output_layer
    while not hasattr(layer, 'input_var'):
        if hasattr(layer, 'input_layer'):
            layer = layer.input_layer
        elif hasattr(layer, 'input_layers'):
            return sum((get_input_vars(l) for l in layer.input_layers), [])
    return [layer.input_var]

class GpuNeuralNet(nolearn.lasagne.NeuralNet):
    '''Like nolearn.lasagne.NeuralNet but then on GPU.

    http://deeplearning.net/tutorial/logreg.html for reference
    '''
    def __init__(
        self,
        layers,
        update=lasagne.updates.nesterov_momentum,
        loss=None,  # BBB
        objective=nolearn.lasagne.objective,
        objective_loss_function=None,
        batch_iterator_train=nolearn.lasagne.BatchIterator(batch_size=128),
        batch_iterator_test=nolearn.lasagne.BatchIterator(batch_size=128),
        regression=False,
        max_epochs=100,
        train_split=nolearn.lasagne.TrainSplit(eval_size=0.2),
        custom_score=None,
        X_tensor_type=None,
        y_tensor_type=None,
        use_label_encoder=False,
        on_batch_finished=None,
        on_epoch_finished=None,
        on_training_started=None,
        on_training_finished=None,
        more_params=None,
        verbose=0,
        **kwargs
    ):
        super(GpuNeuralNet, self).__init__(
            layers,
            update,
            loss,
            objective,
            objective_loss_function,
            batch_iterator_train,
            batch_iterator_test,
            regression,
            max_epochs,
            train_split,
            custom_score,
            X_tensor_type,
            y_tensor_type,
            use_label_encoder,
            on_batch_finished,
            on_epoch_finished,
            on_training_started,
            on_training_finished,
            more_params,
            verbose,
            **kwargs
        )

        old_train_split = self.train_split
        def overwritten_train_split(X, y, self):
            # XXX: remember, X and y are indices!
            # But stratified fold expects y to be class values,
            # so we give those, and then have to reorder y...
            y_class = self.y_shared.get_value()[y]
            X_train, X_valid, y_train, y_valid = old_train_split(X, y_class, self)
            y_train = X_train
            y_valid = X_valid
            return X_train, X_valid, y_train, y_valid
        self.train_split = overwritten_train_split

    def initialize_shared_weights(self, X, y=None):
        # XXX: just going to overwrite current values.
        if X is not None:
            self.X_shared = theano.shared(X)
        if y is not None:
            self.y_shared = theano.shared(y)

    def initialize(self, X=None, y=None):
        assert (X is not None and y is not None) or (X is None and y is None)
        if X is not None:
            self.initialize_shared_weights(X, y)
        super(GpuNeuralNet, self).initialize()

    def fit(self, X, y, epochs=None):
        self.initialize_shared_weights(X, y)
        # XXX: Pass X and y as indices instead of datasets.
        # Now the batch iterator will return slices of indices,
        # will are used to index the shared variables.
        X = np.arange(len(X), dtype=np.int32)
        y = np.arange(len(y), dtype=np.int32)
        return super(GpuNeuralNet, self).fit(X, y, epochs)

    def predict(self, X):
        self.initialize_shared_weights(X)
        X = np.arange(len(X), dtype=np.int32)
        return super(GpuNeuralNet, self).predict(X)

    def predict_proba(self, X):
        self.initialize_shared_weights(X)
        X = np.arange(len(X), dtype=np.int32)
        return super(GpuNeuralNet, self).predict_proba(X)

    def _create_iter_funcs(self, layers, objective, update, output_type):
        y_batch = output_type('y_batch')

        output_layer = layers[-1]
        objective_kw = self._get_params_for('objective')

        loss_train = objective(
            layers, target=y_batch, **objective_kw)
        loss_eval = objective(
            layers, target=y_batch, deterministic=True, **objective_kw)
        predict_proba = lasagne.layers.get_output(output_layer, None, deterministic=True)
        if not self.regression:
            predict = predict_proba.argmax(axis=1)
            accuracy = T.mean(T.eq(predict, y_batch))
        else:
            accuracy = loss_eval

        all_params = self.get_all_params(trainable=True)
        update_params = self._get_params_for('update')
        updates = update(loss_train, all_params, **update_params)

        input_layers = [layer for layer in layers.values()
                        if isinstance(layer, lasagne.layers.InputLayer)]
        assert len(input_layers) == 1, 'Multiple input layers not supported'

        # XXX: Original code wraps all variables in params.
        # For whatever reason Theano does not want this.
        # Not sure what the effects are of not wrapping Variable in Param.
        Xvar = input_layers[0].input_var
        yvar = y_batch

        # I found that passing a vector [1, 2, 3, 4]
        # was much faster than using a slice [1:4+1].
        # This is why I use vectors here, rather than scalars.
        ix, iy = (T.ivector('Xidx'), T.ivector('yidx'))

        train_iter = theano.function(
            inputs=[ix, iy],
            outputs=[loss_train, accuracy],
            updates=updates,
            givens={
                Xvar: self.X_shared[ix],
                yvar: self.y_shared[iy],
            },
        )

        eval_iter = theano.function(
            inputs=[ix, iy],
            outputs=[loss_eval, accuracy],
            givens={
                Xvar: self.X_shared[ix],
                yvar: self.y_shared[iy],
            },
        )

        predict_iter = theano.function(
            inputs=[ix],
            outputs=predict_proba,
            givens={Xvar: self.X_shared[ix]},
        )

        return train_iter, eval_iter, predict_iter
