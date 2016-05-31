'''Deconvolutional networks need to record the position of the pooling switches
during the forward pass. The 'switches' indicate the position of the maximum.

Example:

X = [[1, 2, 3, 4],
     [5, 6, 5, 1],
     [9, 0, 1, 9],
     [3, 4, 5, 6]]

Applying maxpooling with stride = pool_size = (2, 2),
we have the following segments:

X1 = [[1, 2],
      [5, 6]]
X2 = [[3, 4],
      [5, 1]]
X3 = [[9, 0],
      [3, 4]]
X4 = [[1, 9],
      [5, 6]]

For X1 the maximum is 6, local (1, 1), global (1, 1), switch = 5
For X2 the maximum is 5, local (1, 0), global (1, 2), switch = 6
For X3 the maximum is 9, local (0, 0), global (2, 0), switch = 7
For X4 the maximum is 9, local (0, 1), global (2, 3), swtich = 10
'''
import numpy as np
import lasagne.layers
from theano import gof, Op, tensor
import theano

# Renamed between versions
try:
    Pool2DLayer = lasagne.layers.Poold2DLayer
except AttributeError:
    Pool2DLayer = lasagne.layers.MaxPool2DLayer

class MaxSwitch2DLayer(Pool2DLayer):

    def __init__(self, incoming, pool_size, stride=None, pad=(0, 0),
                 ignore_border=True, **kwargs):
        super(MaxSwitch2DLayer, self).__init__(
            incoming,
            pool_size,
            stride,
            pad,
            ignore_border,
            mode='max',
            **kwargs
        )

    def get_output_for(self, input, **kwargs):
        switches = max_switches_2d(
            input,
            ds=self.pool_size,
            st=self.stride,
            ignore_border=self.ignore_border,
            padding=self.pad,
        )
        return switches

def max_switches_2d(
    input, ds, ignore_border=False, st=None, padding=(0, 0)
):
    if input.ndim != 4:
        raise NotImplementedError('max_switches_2d requires a dimension == 4')
    op = DownsampleFactorMaxSwitches(ds, ignore_border, st=st, padding=padding)
    output = op(input)
    return output

class DownsampleFactorMaxSwitches(Op):
    __props__ = ('ds', 'ignore_border', 'st', 'padding')

    @staticmethod
    def out_shape(imgshape, ds, ignore_border=False, st=None, padding=(0, 0)):
        if len(imgshape) != 4:
            raise TypeError('imgshape should be (nbatch, nchan, nrows, ncols)')

        if st is None:
            st = ds
        r, c = imgshape[-2:]
        r += padding[0] * 2
        c += padding[1] * 2

        if ignore_border:
            out_r = (r - ds[0]) // st[0] + 1
            out_c = (c - ds[1]) // st[1] + 1
            if isinstance(r, theano.Variable):
                nr = tensor.maximum(out_r, 0)
            else:
                nr = np.maximum(out_r, 0)
            if isinstance(c, theano.Variable):
                nc = tensor.maximum(out_c, 0)
            else:
                nc = np.maximum(out_c, 0)
        else:
            if isinstance(r, theano.Variable):
                nr = tensor.switch(tensor.ge(st[0], ds[0]),
                                   (r - 1) // st[0] + 1,
                                   tensor.maximum(0, (r - 1 - ds[0])
                                                  // st[0] + 1) + 1)
            elif st[0] >= ds[0]:
                nr = (r - 1) // st[0] + 1
            else:
                nr = max(0, (r - 1 - ds[0]) // st[0] + 1) + 1

            if isinstance(c, theano.Variable):
                nc = tensor.switch(tensor.ge(st[1], ds[1]),
                                   (c - 1) // st[1] + 1,
                                   tensor.maximum(0, (c - 1 - ds[1])
                                                  // st[1] + 1) + 1)
            elif st[1] >= ds[1]:
                nc = (c - 1) // st[1] + 1
            else:
                nc = max(0, (c - 1 - ds[1]) // st[1] + 1) + 1

        rval = list(imgshape[:-2]) + [nr, nc]
        return rval

    def __init__(self, ds, ignore_border=False, st=None, padding=(0, 0)):
        self.ds = tuple(ds)
        if not all([isinstance(d, int) for d in ds]):
            raise ValueError(
                "DownsampleFactorMaxSwitches downsample parameters must be ints."
                " Got %s" % str(ds))
        if st is None:
            st = ds
        assert isinstance(st, (tuple, list))
        self.st = tuple(st)
        self.ignore_border = ignore_border
        self.padding = tuple(padding)
        if self.padding != (0, 0) and not ignore_border:
            raise NotImplementedError(
                'padding works only with ignore_border=True')
        if self.padding[0] >= self.ds[0] or self.padding[1] >= self.ds[1]:
            raise NotImplementedError(
                'padding_h and padding_w must be smaller than strides')

    def make_node(self, x):
        if x.type.ndim != 4:
            raise TypeError()
        x = tensor.as_tensor_variable(x)
        return gof.Apply(self, [x], [x.type()])

    def perform(self, node, inp, out):
        x, = inp
        z, = out
        if len(x.shape) != 4:
            raise NotImplementedError(
                'DownsampleFactorMaxSwitches requires 4D input for now')
        z_shape = self.out_shape(x.shape, self.ds, self.ignore_border, self.st,
                                 self.padding)
        if (z[0] is None) or (z[0].shape != z_shape):
            z[0] = np.empty(z_shape, dtype=x.dtype)
        zz = z[0]
        # number of pooling output rows
        pr = zz.shape[-2]
        # number of pooling output cols
        pc = zz.shape[-1]
        ds0, ds1 = self.ds
        st0, st1 = self.st
        pad_h = self.padding[0]
        pad_w = self.padding[1]
        img_rows = x.shape[-2] + 2 * pad_h
        img_cols = x.shape[-1] + 2 * pad_w

        # pad the image
        if self.padding != (0, 0):
            y = np.zeros(
                (x.shape[0], x.shape[1], img_rows, img_cols),
                dtype=x.dtype)
            y[:, :, pad_h:(img_rows-pad_h), pad_w:(img_cols-pad_w)] = x
        else:
            y = x

        for n in range(x.shape[0]):
            for k in range(x.shape[1]):
                for r in range(pr):
                    row_st = r * st0
                    row_end = min(row_st + ds0, img_rows)
                    for c in range(pc):
                        col_st = c * st1
                        col_end = min(col_st + ds1, img_cols)
                        zz[n, k, r, c] = np.argmax(
                            y[n, k, row_st:row_end, col_st:col_end])

    def infer_shape(self, node, in_shapes):
        shp = self.out_shape(in_shapes[0], self.ds,
                             self.ignore_border, self.st, self.padding)
        return [shp]

    def c_headers(self):
        return ['<algorithm>']

    def c_code(self, node, name, inp, out, sub):
        x, = inp
        z, = out
        fail = sub['fail']
        ignore_border = int(self.ignore_border)
        ds0, ds1 = self.ds
        st0, st1 = self.st
        pd0, pd1 = self.padding
        ccode = """
        int z_r, z_c; // shape of the output
        int r, c; // shape of the padded_input
        if(PyArray_NDIM(%(x)s)!=4)
        {
            PyErr_SetString(PyExc_ValueError, "x must be a 4d ndarray");
            %(fail)s;
        }
        r = PyArray_DIMS(%(x)s)[2];
        c = PyArray_DIMS(%(x)s)[3];
        r += %(pd0)s * 2;
        c += %(pd1)s * 2;

        if (%(pd0)s != 0 && %(pd1)s != 0 && !%(ignore_border)s)
            {
              PyErr_SetString(PyExc_ValueError,
                "padding must be (0,0) when ignore border is False");
              %(fail)s;
            }
        if (%(ignore_border)s)
        {

            // '/' in C is different from '/' in python
            if (r - %(ds0)s < 0)
            {
              z_r = 0;
            }
            else
            {
              z_r = (r - %(ds0)s) / %(st0)s + 1;
            }
            if (c - %(ds1)s < 0)
            {
              z_c = 0;
            }
            else
            {
              z_c = (c - %(ds1)s) / %(st1)s + 1;
            }
        }
        else
        {
            // decide how many rows the output has
            if (%(st0)s >= %(ds0)s)
            {
                z_r = (r - 1) / %(st0)s + 1;
            }
            else
            {
                z_r = std::max(0, (r - 1 - %(ds0)s) / %(st0)s + 1) + 1;
            }
            // decide how many columns the output has
            if (%(st1)s >= %(ds1)s)
            {
                z_c = (c - 1) / %(st1)s + 1;
            }
            else
            {
                z_c = std::max(0, (c - 1 - %(ds1)s) / %(st1)s + 1) + 1;
            }
        }
        // memory allocation of z if necessary
        if ((!%(z)s)
          || *PyArray_DIMS(%(z)s)!=4
          ||(PyArray_DIMS(%(z)s)[0] != PyArray_DIMS(%(x)s)[0])
          ||(PyArray_DIMS(%(z)s)[1] != PyArray_DIMS(%(x)s)[1])
          ||(PyArray_DIMS(%(z)s)[2] != z_r)
          ||(PyArray_DIMS(%(z)s)[3] != z_c)
          )
        {
          if (%(z)s) Py_XDECREF(%(z)s);
          npy_intp dims[4] = {0,0,0,0};
          dims[0]=PyArray_DIMS(%(x)s)[0];
          dims[1]=PyArray_DIMS(%(x)s)[1];
          dims[2]=z_r;
          dims[3]=z_c;
          int typenum = PyArray_ObjectType((PyObject*)%(x)s, 0);
          %(z)s = (PyArrayObject*) PyArray_ZEROS(4, dims, typenum,0);
          //%(z)s = (PyArrayObject*) PyArray_Zeros(4, dims, PyArray_ObjectType((PyObject*)%(x)s, 0), 0);
          // %(z)s = (PyArrayObject*) PyArray_Empty(4, dims, PyArray_DescrNewFromType(NPY_LONG), 0);
        }

        // used for indexing a pool region inside the input
        int r_st, r_end, c_st, c_end;
        dtype_%(x)s value_collector; // temp var for the value in a region
        dtype_%(z)s switch_collector; // temp var for the switch in a region
        if (z_r && z_c)
        {
            for(int b=0; b<PyArray_DIMS(%(x)s)[0]; b++){
              for(int k=0; k<PyArray_DIMS(%(x)s)[1]; k++){
                for(int i=0; i< z_r; i++){
                  r_st = i * %(st0)s;
                  r_end = r_st + %(ds0)s;
                  // skip the padding
                  r_st = r_st < %(pd0)s ? %(pd0)s : r_st;
                  r_end = r_end > (r - %(pd0)s) ? r - %(pd0)s : r_end;
                  // from padded_img space to img space
                  r_st -= %(pd0)s;
                  r_end -= %(pd0)s;

                  // handle the case where no padding, ignore border is True
                  if (%(ignore_border)s)
                  {
                    r_end = r_end > r ? r : r_end;
                  }
                  for(int j=0; j<z_c; j++){
                    c_st = j * %(st1)s;
                    c_end = c_st + %(ds1)s;
                    // skip the padding
                    c_st = c_st < %(pd1)s ? %(pd1)s : c_st;
                    c_end = c_end > (c - %(pd1)s) ? c - %(pd1)s : c_end;
                    dtype_%(z)s * z = (
                          (dtype_%(z)s*)(PyArray_GETPTR4(%(z)s, b, k, i, j)));
                    // change coordinates from padding_img space into img space
                    c_st -= %(pd1)s;
                    c_end -= %(pd1)s;
                    // handle the case where no padding, ignore border is True
                    if (%(ignore_border)s)
                    {
                      c_end = c_end > c ? c : c_end;
                    }

                    // use the first element as the initial value of value_collector
                    value_collector = ((dtype_%(x)s*)(PyArray_GETPTR4(%(x)s,b,k,r_st,c_st)))[0];
                    switch_collector = 0;
                    // go through the pooled region in the unpadded input
                    for(int m=r_st; m<r_end; m++)
                    {
                      for(int n=c_st; n<c_end; n++)
                      {
                        dtype_%(x)s a = ((dtype_%(x)s*)(PyArray_GETPTR4(%(x)s,b,k,m,n)))[0];
                        if (a > value_collector) {
                            value_collector = a;
                            switch_collector = (m - r_st) * z_r + (n - c_st);
                        }
                      }
                    }
                    z[0] = switch_collector;
                  }
                }
              }
            }
        }
        """
        return ccode % locals()

    def c_code_cache_version(self):
        return (0, 6, 8, 3)
