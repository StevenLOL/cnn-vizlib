'''
'''
import vizlib
import theano
import numpy as np

def occlusion_multi_input(xs, output_layer, output_node, square_length=7, ignore_nonlinearity=False):
    '''Like occlusion, but expects multiple inputs.
    Only the first input is occluded.
    '''
    X, X2 = xs

    if (X.ndim != 4) or X.shape[0] != 1:
        raise ValueError("This function requires the input data to be of "
                         "shape (1, c, x, y), instead got {}".format(X.shape))
    if square_length % 2 == 0:
        raise ValueError("Square length has to be an odd number, instead "
                         "got {}.".format(square_length))

    img = X[0].copy()
    bs, col, s0, s1 = X.shape

    saliency_map = np.zeros((s0, s1))
    pad = square_length // 2
    x_occluded = np.zeros((1, col, s0, s1), dtype=img.dtype)

    X_vars = vizlib.utils.get_input_vars(output_layer)
    scores = vizlib.activation_maximization.scores(
        X_vars, output_layer, ignore_nonlinearity=ignore_nonlinearity)
    predict_proba = theano.function(X_vars, scores[output_node])

    # generate occluded images
    x_occluded[0]
    base_likelihood = predict_proba(X, X2)
    for i in range(s0):
        for j in range(s1):
            if pad == 0:
                x_pad = img.copy()
                x_pad[:, i, j] = 0.
                x_occluded[0] = x_pad
            else:
                x_pad = np.pad(img, ((0, 0), (pad, pad), (pad, pad)), 'constant')
                x_pad[:, i:i + square_length, j:j + square_length] = 0.
                x_occluded[0] = x_pad[:, pad:-pad, pad:-pad]

            # A pixel is more salient if it causes a large drop in the
            # probability, i.e., base_likelihood >> predict_proba
            saliency_map[i, j] = base_likelihood - predict_proba(x_occluded, X2)

    return saliency_map



def occlusion(X, output_layer, square_length=7, ignore_nonlinearity=False):
    # The following was largely implemented by nolearn.
    # I just had to change some things around to accept a single output_layer,
    # rather than a neural net, and made it use batches instead of single inputs.
    if (X.ndim != 4):
        raise ValueError("This function requires the input data to be of "
                         "shape (b, c, x, y), instead got {}".format(X.shape))
    if square_length % 2 == 0:
        raise ValueError("Square length has to be an odd number, instead "
                         "got {}.".format(square_length))

    X_var = vizlib.utils.get_input_var(output_layer)
    scores = vizlib.utils.get_output(output_layer, X_var, ignore_nonlinearity=ignore_nonlinearity)
    predict_proba = theano.function([X_var], scores)
    base_likelihood = predict_proba(X)
    num_outputs = base_likelihood.shape[1]

    img = X.copy()
    bs, col, s0, s1 = X.shape
    saliency_map = np.zeros((bs, num_outputs, s0, s1))
    pad = square_length // 2
    x_occluded = np.zeros((bs, col, s0, s1), dtype=img.dtype)

    for i in range(s0):
        for j in range(s1):
            if pad == 0:
                x_pad = img.copy()
                x_pad[:, :, i, j] = 0.
                x_occluded = x_pad
            else:
                x_pad = np.pad(img, ((0, 0), (0, 0), (pad, pad), (pad, pad)), 'constant')
                x_pad[:, i:i + square_length, j:j + square_length] = 0.
                x_occluded = x_pad[:, :, pad:-pad, pad:-pad]

            # A pixel is more salient if it causes a large drop in the
            # probability, i.e., base_likelihood >> predict_proba
            saliency_map[:, :, i, j] = base_likelihood - predict_proba(x_occluded)

    return saliency_map
