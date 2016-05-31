'''
'''
import numpy as np
import math

__all__ = ['occlusion', 'occlusion_iter']

class OcclusionBatchIterator(object):

    def __init__(self, xs, batch_slice, stride, square_length, pad_value):
        bs, col, s0, s1 = xs.shape
        if batch_slice is None:
            batch_slice = slice(0, bs)

        self.img = xs.copy()
        self.pad = square_length // 2
        self.square_length = square_length
        self.stride = stride
        self.pad_value = pad_value
        self.batch_slice = batch_slice

    def __len__(self):
        return (((self.img.shape[2] - 1) // self.stride + 1)
                * ((self.img.shape[3] - 1) // self.stride + 1))

    def __iter__(self):
        stride = self.stride

        for i in range(0, self.img.shape[2], stride):
            for j in range(0, self.img.shape[3], stride):
                yield (i, j, self.occlude(i, j))

    def __getitem__(self, index):
        return self.occlude(*index)

    def occlude(self, i, j):
        pad = self.pad
        pad_value = self.pad_value
        square_length = self.square_length
        batch_slice = self.batch_slice
        img = self.img

        if pad == 0:
            x_pad = self.img.copy()
            x_pad[batch_slice, :, i, j] = pad_value
            x_occluded = x_pad
        else:
            x_pad = np.pad(img, ((0, 0), (0, 0), (pad, pad), (pad, pad)), 'constant')
            x_pad[batch_slice, :, i:i + square_length, j:j + square_length] = pad_value
            x_occluded = x_pad[:, :, pad:-pad, pad:-pad]

        return x_occluded


def occlusion_iter(xs, batch_slice=None, stride=1, square_length=7, pad_value=0.0):
    '''Returns an iterator that occludes the entries in xs.

    Assumes
        xs.shape = (batch, chan, width, height)

    Occludes an entire batch at once.

    If batch_slice is an integer then only this batch entry is occluded.
    If batch_slice is an array of indices then these batch entries are occluded.
    if batch_slice is none all batch entries are occluded.
    '''
    if (xs.ndim != 4):
        raise ValueError("This function requires the input data to be of "
                         "shape (b, c, x, y), instead got {}".format(xs.shape))
    if square_length % 2 == 0:
        raise ValueError("Square length has to be an odd number, instead "
                         "got {}.".format(square_length))
    if stride > square_length:
        raise ValueError("Stride should be LEQ to the square_length")

    return OcclusionBatchIterator(xs, batch_slice, stride, square_length, pad_value)

def occlusion(iter_batch_or_xs, predict_proba, **kwargs):
    if not isinstance(iter_batch_or_xs, OcclusionBatchIterator):
        iter_batch_or_xs = occlusion_iter(iter_batch_or_xs, **kwargs)

    base_likelihood = predict_proba(iter_batch_or_xs.img)
    bs, c, s0, s1 = iter_batch_or_xs.img.shape
    num_outputs = base_likelihood.shape[1]
    saliency_map = np.zeros((bs, num_outputs, s0, s1))

    total = len(iter_batch_or_xs)
    shown_percent = 0
    for p, (i, j, x_occluded) in enumerate(iter_batch_or_xs):
        saliency_map[:, :, i, j] = base_likelihood - predict_proba(x_occluded)

        progress = float(p) / total * 100
        if math.floor(progress / 10) > shown_percent:
            print('Progress: ', progress)
            shown_percent = math.floor(progress / 10)

    return saliency_map
