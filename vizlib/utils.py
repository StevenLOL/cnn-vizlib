'''
'''
import nolearn.lasagne
import lasagne
import theano
import theano.tensor as T
import numpy as np
from time import time
from collections import OrderedDict

class IgnoreNonlinearity():
    def __init__(self, layer, ignore_nonlinearity):
        self.layer = layer
        self.nonlinearity = None
        self.ignore_nonlinearity = ignore_nonlinearity

    def __enter__(self):
        if self.ignore_nonlinearity:
            self.nonlinearity = self.layer.nonlinearity
            self.layer.nonlinearity = lasagne.nonlinearities.identity

    def __exit__(self, type, value, traceback):
        if self.ignore_nonlinearity:
            self.layer.nonlinearity = self.nonlinearity

def get_input_var(output_layer):
    layer = output_layer
    while not hasattr(layer, 'input_var'):
        layer = layer.input_layer
    return layer.input_var

def get_input_vars_dict(output_layer):
    result = OrderedDict()
    layer = output_layer
    while not hasattr(layer, 'input_var'):
        if hasattr(layer, 'input_layer'):
            layer = layer.input_layer
        elif hasattr(layer, 'input_layers'):
            for l in layer.input_layers:
                result.update(get_input_vars_dict(l))
            return result
    return OrderedDict([(layer, layer.input_var)])

def get_input_vars(output_layer):
    layer = output_layer
    while not hasattr(layer, 'input_var'):
        if hasattr(layer, 'input_layer'):
            layer = layer.input_layer
        elif hasattr(layer, 'input_layers'):
            return sum((get_input_vars(l) for l in layer.input_layers), [])
    return [layer.input_var]

def get_input_layers(output_layer):
    return [
        l
        for l in lasagne.layers.get_all_layers(output_layer)
        if isinstance(l, lasagne.layers.InputLayer)
    ]

def get_output_expressions(X, output_layer):
    '''Return a list of the output expressions wrt the input variable X
    for all layers that precide the output_layer and the output_layer itself.

    i.e., if output_layer has 9 layers before it,
    then this will return a list x with len(x) == 10,
    where x[-1] is the output expression of the output_layer
    '''

    # We assume layers have a single input and a single output,
    # so building the computation graph is as simple as traversing
    # to the input, and then traversing back up.
    first_layer = output_layer
    layers = [output_layer]
    while not hasattr(first_layer, 'input_var'):
        first_layer = first_layer.input_layer
        layers.append(first_layer)
    # the pop removes the first_layer
    layers.pop()
    output_expr = X
    layers.reverse()
    expressions = [output_expr]

    for current_layer in layers:
        output_expr = current_layer.get_output_for(output_expr, deterministic=True)
        expressions.append(output_expr)

    return expressions

def get_output(output_layer, input, ignore_nonlinearity=False):
    if ignore_nonlinearity:
        original_nonlinearity = output_layer.nonlinearity
    output_layer.nonlinearity = lasagne.nonlinearities.identity

    output = get_output_expressions(input, output_layer)[-1]

    if ignore_nonlinearity:
        output_layer.nonlinearity = original_nonlinearity

    return output

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


def export_taylor_features(xs, feature_functions, fname):
    '''
    '''
    def filter_batch(arr):
        temp = []
        for i in xrange(arr.shape[1]):
            temp.append(arr[i, i, :, :, :])
        return np.array(temp)

    max_maps = []
    sum_maps = []
    for pl in feature_functions:
        max_maps.append([])
        sum_maps.append([])
        for f in pl:
            max_maps[-1].append([])
            sum_maps[-1].append([])
            for x in xs:
                max_output, sum_output = f(x[0])
                max_output = filter_batch(max_output)
                sum_output = filter_batch(sum_output)
                max_maps[-1][-1].append(max_output.max(axis=1))
                sum_maps[-1][-1].append(sum_output.max(axis=1)) # max here is not an error -- sum refers to the gradient
        max_maps[-1] = np.array(max_maps[-1])
        sum_maps[-1] = np.array(sum_maps[-1])
    # max_maps are jagged length --- not an equal number of features for each layer
    # so keep first dimension a list, rest numpy arrays
    np.savez(fname.format('max'), *max_maps)
    np.savez(fname.format('sum'), *sum_maps)

    return max_maps, sum_maps

def export_output_features(xs, output_functions, fname):
    maps_img_x_batch = []
    for x in xs:
        maps_img = []
        for f in output_functions:
            map = f(*x)
            maps_img.append(map)
        maps_img_x_batch.append(maps_img)

    maps_arr = np.array(maps_img_x_batch)

    # image x batch x output x batch x c x 0 x 1
    # reason for 2x batch:
    # output of first eye is also dependent on second eye and vice versa
    # can simply ignore the last batch dimension and take the i-th slice,
    # where i is the index into the first batch dimension.
    maps_arr = maps_arr.max(axis=-3)
    temp = []
    for i in xrange(maps_arr.shape[1]):
        temp.append(maps_arr[:, i, :, i, :, :])
    maps_arr = np.array(temp)

    np.save(fname, maps_arr)
    return maps_arr

def receptive_field_size(layers):
    if len(layers) == 0:
        return 0
    layers = list(reversed(layers))
    fs = layers[0][0]
    for ks, stride in layers[1:]:
        fs = (fs - 1) * stride + ks
    return fs

def receptive_field_size_for_layer(layer):
    layers = lasagne.layers.get_all_layers(layer)
    return receptive_field_size(
        [(kernel_size(l), stride(l)) for l in layers if is_pool_or_conv_layer(l)]
    )

def stride(l):
    assert l.stride[0] == l.stride[1], 'not implemented'
    return l.stride[0]

def kernel_size(l):
    try:
        ks = l.filter_size
    except:
        ks = l.pool_size
    assert ks[0] == ks[1], 'not implemented'
    return ks[0]

def is_pool_or_conv_layer(l):
    return hasattr(l, 'filter_size') or hasattr(l, 'pool_size')
