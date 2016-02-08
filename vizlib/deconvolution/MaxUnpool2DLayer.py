'''
'''
import lasagne
import theano
import numpy as np

from .MaxSwitch2DLayer import MaxSwitch2DLayer

class MaxUnpool2DLayer(lasagne.layers.MergeLayer):

    def __init__(self, incomings, **kwargs):
        super(MaxUnpool2DLayer, self).__init__(incomings, **kwargs)

    def get_output_for(self, inputs, **kwargs):
        op = UpsampleFactorMax(
            self.input_layers[0].input_shape,
            self.input_layers[0].pool_size,
        )
        output = op(*inputs)
        return output

    def get_output_shape_for(self, input_shapes):
        return self.input_layers[0].input_shape


class UpsampleFactorMax(theano.Op):
    __props__ = ('output_shape', 'pool_size')

    def __init__(self, output_shape, pool_size):
        super(UpsampleFactorMax, self).__init__()
        self.output_shape = output_shape
        self.pool_size = pool_size

    def make_node(self, pooled, switches):
        pooled = theano.tensor.as_tensor_variable(pooled)
        switches = theano.tensor.as_tensor_variable(switches)
        return theano.Apply(self, [pooled, switches], [pooled.type()])

    def perform(self, node, inputs, output_storage):
        pooled, switches = inputs
        output = output_storage[0]

        output_shape = list(self.output_shape)
        output_shape[0] = pooled.shape[0]

        if (output[0] is None) or (output[0].shape != output_shape):
            output[0] = np.empty(output_shape, dtype=pooled.dtype)
        output = output[0]

        w, h = self.pool_size
        b, c, nr, nc = pooled.shape
        for i in range(b):
            for j in range(c):
                for k in range(nr):
                    for l in range(nc):
                        # translate localized id indices to global 2d
                        s = int(switches[i, j, k, l])
                        rk = k * w + s // w
                        rl = l * h + s % w
                        output[i, j, rk, rl] = pooled[i, j, k, l]


    # def c_code(self, node, inp, out, sub):
    #     pass

    # def c_code_cache_version(self):
    #     return None
