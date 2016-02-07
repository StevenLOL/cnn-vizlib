import numpy as np


def unpool(
    pooled_input, switches,
    pool_size,
    stride=None,
    padding=(0, 0)
):
    if stride is None:
        stride = pool_size
    if stride != pool_size:
        raise ValueError()
    if padding != (0, 0):
        raise ValueError()

    shape = list(pooled_input.shape)
    w, h = pool_size
    shape[-2] = pooled_input.shape[-2] * h
    shape[-1] = pooled_input.shape[-1] * w

    result = np.zeros(tuple(shape), dtype=pooled_input.dtype)

    b, c, nr, nc = pooled_input.shape
    for i in xrange(b):
        for j in xrange(c):
            for k in xrange(nr):
                for l in xrange(nc):
                    # translate localized id indices to global 2d
                    s = int(switches[i, j, k, l])
                    rk = k * w + s // w
                    rl = l * h + s % w
                    result[i, j, rk, rl] = pooled_input[i, j, k, l]

    return result
