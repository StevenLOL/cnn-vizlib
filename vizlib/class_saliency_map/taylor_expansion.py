'''Implement class saliency visualization as per [1]

The key idea here is that we want to rank the pixels of an image based on their
influence on the class scores.

If the class scores were a linear function of the input pixels this would be
easy:

    S_c(I) = w_{c}^{t}I + b_c

The weights would indicate how each pixel should be ranked.

We can create a locally linear approximation of S_c using taylor series
expansion, by defining w as:

    w = dS_c/dI | I_0

Here I_0 is the image to evaluate at.

[1] simonyan2013visualizing.pdf
'''
import vizlib
import theano
import lasagne
from vizlib.utils import IgnoreNonlinearity
import numpy as np
import time


def taylor_expansion_output_layer(output_layer, ignore_nonlinearity=False):
    '''Returns a list of functions dSi / dX, where X is the batch and Si is the output.

    This is assuming the features in the batch are dependent,
    and that the variable we are interested in is the first variable.

    For the Jeffrey network this means we should set the batch size to 2.

    Returns a list of theano functions `fs` of `len(fs) == n_batch`,
    where `fs[i](x).shape == n_batch x n_output x c x 0 x 1`
    '''
    output_shape = output_layer.output_shape
    n_batch = output_shape[0]
    if n_batch is None:
        print('Expecting batch size to be fixed, assuming 1')
        n_batch = 1

    X_vars_dict = vizlib.utils.get_input_vars_dict(output_layer)
    X_vars = X_vars_dict.values()

    with IgnoreNonlinearity(output_layer, ignore_nonlinearity):
        output_expr = lasagne.layers.get_output(output_layer, deterministic=True)

    function_per_one = []
    for i in xrange(n_batch):
        output = output_expr[i, :]
        f = theano.function(X_vars, theano.gradient.jacobian(output, wrt=X_vars[0]))
        print(time.strftime('%H:%M:%S'), float(i) / n_batch)
        function_per_one.append(f)
    return function_per_one

def taylor_expansion_feature_layer(output_layer, ignore_nonlinearity=False):
    '''Returns a list of functions fs, one for each feature in the feature layer.

    len(fs) == output_layer.shape[1]

    Computes the gradient of the max and the sum of the feature.
    len(fs[i](x)) == 2
    fs[i](x)[0] == max
    fs[i](x)[1] == sum
    '''
    output_shape = output_layer.output_shape
    n_features = output_shape[1]

    X_vars_dict = vizlib.utils.get_input_vars_dict(output_layer)
    X_vars = X_vars_dict.values()

    with vizlib.utils.IgnoreNonlinearity(output_layer, ignore_nonlinearity):
        output_expr = lasagne.layers.get_output(output_layer, deterministic=True)

    function_per_feature = []
    for i in xrange(n_features):
        f = theano.function(X_vars, [
                theano.gradient.jacobian(
            output_expr[:, i].max(axis=(1, 2)), wrt=X_vars[0]
        ),
        theano.gradient.jacobian(
            output_expr[:, i].sum(axis=(1, 2)), wrt=X_vars[0]
        )])
        print(time.strftime('%H:%M:%S'), float(i) / n_features)
        function_per_feature.append(f)
    return function_per_feature
