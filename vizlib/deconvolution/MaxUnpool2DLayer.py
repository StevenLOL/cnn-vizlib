'''
'''
import lasagne
import theano
import numpy as np

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
            output[0] = np.zeros(output_shape, dtype=pooled.dtype)
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


    def c_code(self, node, name, inputs, output_storage, sub):
        pooled, switches = inputs
        output = output_storage[0]

        output_shape = list(self.output_shape)
        _, _, onr, onc = output_shape

        w, h = self.pool_size

        ccode = """
        int b = PyArray_DIMS(%(pooled)s)[0];
        int c = PyArray_DIMS(%(pooled)s)[1];
        // input number of rows
        int nr = PyArray_DIMS(%(pooled)s)[2];
        int nc = PyArray_DIMS(%(pooled)s)[3];
        // output number of rows
        int onr = %(onr)d;
        int onc = %(onc)d;
        // memory allocation of z if necessary
        if ((!%(output)s)
          || *PyArray_DIMS(%(output)s)!=4
          ||(PyArray_DIMS(%(output)s)[0] != b)
          ||(PyArray_DIMS(%(output)s)[1] != c)
          ||(PyArray_DIMS(%(output)s)[2] != onr)
          ||(PyArray_DIMS(%(output)s)[3] != onc)
          )
        {
          if (%(output)s) Py_XDECREF(%(output)s);
          npy_intp dims[4] = {0,0,0,0};
          dims[0]=b;
          dims[1]=c;
          dims[2]=onr;
          dims[3]=onc;
          int typenum = PyArray_ObjectType((PyObject*)%(pooled)s, 0);
          %(output)s = (PyArrayObject*) PyArray_ZEROS(4, dims, typenum,0);
        }

        int w = %(w)d;
        int h = %(h)d;
        int s, rk, rl;

        dtype_%(pooled)s* p = (dtype_%(pooled)s*) (PyArray_GETPTR4(%(pooled)s,0,0,0,0));
        dtype_%(switches)s* sp = (dtype_%(switches)s*) (PyArray_GETPTR4(%(switches)s,0,0,0,0));

        for (int i = 0; i < b; i++) {
          for (int j = 0; j < c; j++) {
            for (int k = 0; k < nr; k++) {
              for (int l = 0; l < nc; l++) {
                s = *sp;

                rk = k * w + s / w;
                rl = l * h + s %% w;

                dtype_%(output)s* out = (
                    (dtype_%(output)s*)
                    (PyArray_GETPTR4(%(output)s, i, j, rk, rl)));
                *out = *p;

                p++;
                sp++;
              }
            }
          }
        }
        """
        return ccode % locals()

    def c_code_cache_version(self):
        return (0, 6, 8, 3)
