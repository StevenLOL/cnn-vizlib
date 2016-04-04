'''
'''
from . import Deconv2DLayer, MaxSwitch2DLayer, MaxUnpool2DLayer
import lasagne

def get_deconv_output(output_layer, input_var=None):

    top_order = lasagne.layers.get_all_layers(output_layer)
    if input_var is None:
        input_var = top_order[0].input_var

    # find the expressions up to (exclusive) the first DenseLayer
    expressions = [input_var]
    first_dense_idx = 1
    for layer in top_order[1:]:
        if isinstance(layer, lasagne.layers.DenseLayer):
            break
        expressions.append(layer.get_output_for(expressions[-1]))
        first_dense_idx += 1
    expr = expressions[-1]

    # traverse down, building the deconv network
    for i, layer in reversed(list(enumerate(top_order[1:first_dense_idx], 1))):
        if isinstance(layer, lasagne.layers.Conv2DLayer):
            conv_layer = Deconv2DLayer(layer)
            expr = conv_layer.get_output_for(expr)
        if isinstance(layer, lasagne.layers.MaxPool2DLayer):
            switch_layer = MaxSwitch2DLayer(layer.input_layer, layer.pool_size)
            unpool_layer = MaxUnpool2DLayer([layer, switch_layer])
            pool_input_expr = expressions[i-1]
            expr = unpool_layer.get_output_for([expr, pool_input_expr])
    return expr
