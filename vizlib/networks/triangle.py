import lasagne

def build_network(input_shape):
    input = lasagne.layers.InputLayer((None, ) + input_shape)
    conv1 = lasagne.layers.Conv2DLayer(input, 1, (3, 3))
    out = lasagne.layers.DenseLayer(conv1, 2, nonlinearity=lasagne.nonlinearities.softmax)
    return out
