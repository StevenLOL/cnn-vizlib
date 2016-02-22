'''
'''
import nolearn.lasagne
import lasagne
import theano
import theano.tensor as T
import numpy as np


def get_input_var(output_layer):
    layer = output_layer
    while not hasattr(layer, 'input_var'):
        layer = layer.input_layer
    return layer.input_var

class GpuNeuralNet(nolearn.lasagne.NeuralNet):
    '''Like nolearn.lasagne.NeuralNet but then on GPU. '''
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

    def fit(self, X, y, epochs=None):
        self.X_shared = theano.shared(X)
        self.y_shared = theano.shared(y)
        assert not getattr(self, '_initialized', False), \
                "Do not call initialize yourself! It needs self.X_shared, self.y_shared available"

        # XXX: Pass X and y as indices instead of datasets.
        # Now the batch iterator will return slices of indices,
        # will are used to index the shared variables.
        X = np.arange(len(X), dtype=theano.config.floatX)
        y = np.arange(len(y), dtype=np.int32)
        return super(GpuNeuralNet, self).fit(X, y, epochs)

    def _create_iter_funcs(self, layers, objective, update, output_type):
        y_batch = output_type('y_batch')

        output_layer = layers[-1]
        objective_kw = self._get_params_for('objective')

        loss_train = objective(
            layers, target=y_batch, **objective_kw)
        loss_eval = objective(
            layers, target=y_batch, deterministic=True, **objective_kw)
        predict_proba = lasagne.get_output(output_layer, None, deterministic=True)
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
        batch_idxs = [T.ivector('Xidx'), T.ivector('yidx')]
        ix, iy = batch_idxs

        train_iter = theano.function(
            inputs=batch_idxs,
            outputs=[loss_train, accuracy],
            updates=updates,
            givens={Xvar: self.X_shared[ix], yvar: self.y_shared[iy]},
            allow_input_downcast=True,
        )

        eval_iter = theano.function(
            inputs=batch_idxs,
            outputs=[loss_eval, accuracy],
            givens={Xvar: self.X_shared[ix], yvar: self.y_shared[iy]},
            allow_input_downcast=True,
        )

        predict_iter = theano.function(
            inputs=batch_idxs[0:-1],
            outputs=predict_proba,
            givens={Xvar: self.X_shared[ix]},
            allow_input_downcast=True,
        )

        return train_iter, eval_iter, predict_iter
