'''
'''
import vizlib
import theano
import numpy as np


def occlusion(X, output_layer, output_node, square_length=7):
    # The following was largely implemented by nolearn.
    # I just had to change some things around to accept a single output_layer,
    # rather than a neural net
    if (X.ndim != 4) or X.shape[0] != 1:
        raise ValueError("This function requires the input data to be of "
                         "shape (1, c, x, y), instead got {}".format(X.shape))
    if square_length % 2 == 0:
        raise ValueError("Square length has to be an odd number, instead "
                         "got {}.".format(square_length))

    img = X[0].copy()
    bs, col, s0, s1 = X.shape

    saliency_map = np.zeros((s0, s1))
    pad = square_length // 2 + 1
    x_occluded = np.zeros((1, col, s0, s1), dtype=img.dtype)

    X_var = vizlib.utils.get_input_var(output_layer)
    scores = vizlib.activation_maximization.scores(
        X_var, output_layer, ignore_nonlinearity=False)
    predict_proba = theano.function([X_var], scores[output_node])

    # generate occluded images
    for i in range(s0):
        for j in range(s1):
            x_pad = np.pad(img, ((0, 0), (pad, pad), (pad, pad)), 'constant')
            x_pad[:, i:i + square_length, j:j + square_length] = 0.
            x_occluded[0] = x_pad[:, pad:-pad, pad:-pad]
            saliency_map[i, j] = 1 - predict_proba(x_occluded)

    return saliency_map