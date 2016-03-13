'''
'''
import numpy as np
import cv2
import theano
from helpers import DataSet

THICKNESS = 2
OFFSET = 0.1

def circle(size):
    w, h = size
    offset = 2 * OFFSET

    center = (w / 2, h / 2)
    radius = min(int(center[0] * (1 - offset)), int(center[1] * (1-offset)))

    background = np.zeros(size, dtype=theano.config.floatX)
    cv2.circle(background, center, radius, (1, ), THICKNESS)

    return background


def triangle(size):
    w, h = size
    offset = OFFSET

    topcenter = (int(0.5 * w), int(offset * h))
    bottomleft = (int(offset * w), int((1 - offset) * h))
    bottomright = (int((1 - offset) * w), int((1 - offset) * h))
    pts = np.array([topcenter, bottomleft, bottomright])

    background = np.zeros(size, dtype=theano.config.floatX)
    cv2.polylines(background, [pts], 1, (1,), THICKNESS)

    return background

def triangle_and_circle(size=(32, 32)):

    # Most optimizers do not play nicely with only 2 examples,
    # because they split in a train and test set.
    # Here we use some duplication to get around this.
    n = 50
    c = circle(size)
    t = triangle(size)

    X = np.array([c, t])
    X = X[:, None, :, :].repeat(n, axis=0)
    y = np.array([0] * n + [1] * n)

    return DataSet(X, y).shuffle().standardize()
