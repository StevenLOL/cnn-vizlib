#!/usr/bin/env python2.7
# encoding: utf-8
'''
'''
import matplotlib
matplotlib.use('agg')
import vizlib
import tempfile
import numpy as np
import subprocess
import nolearn.lasagne
import lasagne
import cv2
import matplotlib.pyplot as plt


def create_rgb_images(width=9, height=9):
    black = np.zeros((width, height, 3), dtype=np.float32)
    white = np.ones((width, height, 3), dtype=np.float32)
    red = black.copy()
    red[:, :, 0] = 1
    green = black.copy()
    green[:, :, 1] = 1
    blue = black.copy()
    blue[:, :, 2] = 1
    return black, white, red, green, blue


def test_from_numpy_arrays():
    black, white, red, green, blue = create_rgb_images()

    numpy_arrays = [
        black,
        white,
        red,
        green,
        blue,
        green,
        red,
        white,
        black
    ]

    fd, fname = tempfile.mkstemp(suffix='.gif')
    vizlib.gifs.from_numpy_arrays(numpy_arrays, output_file=fname, delay=100)
    subprocess.call(['eog', fname])
    # No asserts, just making sure it doesn't fail.

def test_from_activations():
    black, white, red, green, blue = create_rgb_images()

    activations = np.array([
        [black, white, red],
        [green, blue, green],
        [red, white, black],
    ])
    # We expect the animation to play exactly as written above here,
    # with each row being a frame.

    exports = (
        (vizlib.gifs.from_activations, '.gif', dict(delay=100)),
        (vizlib.animations.from_activations, '.mp4', dict(fps=5)),
    )

    for f, suffix, kwargs in exports:
        fd, fname = tempfile.mkstemp(suffix=suffix)
        f(activations, output_file=fname, **kwargs)
        subprocess.call(['gnome-open', fname])

def test_from_neural_network():
    nn = nolearn.lasagne.NeuralNet(
        layers=[
            ('input', lasagne.layers.InputLayer),
            ('conv1', lasagne.layers.Conv2DLayer),
            ('pool1', lasagne.layers.MaxPool2DLayer),
            ('conv2', lasagne.layers.Conv2DLayer),
            ('pool2', lasagne.layers.MaxPool2DLayer),
            ('hidden4', lasagne.layers.DenseLayer),
            ('output', lasagne.layers.DenseLayer),
        ],
        input_shape=(None, 3, 32, 32),
        # input_shape=(None, 1, 32, 32),
        conv1_num_filters=3, conv1_filter_size=(5, 5), conv1_pad='same',
        pool1_pool_size=(2, 2),
        conv2_num_filters=2, conv2_filter_size=(3, 3), conv2_pad='same',
        pool2_pool_size=(2, 2),
        hidden4_num_units=10,
        output_num_units=6,
        output_nonlinearity=lasagne.nonlinearities.softmax,

        update_learning_rate=0.01,
        update_momentum=0.9,
        regression=False,
        max_epochs=10,
        verbose=1,
    )
    nn.initialize()

    black, white, red, green, blue = create_rgb_images(32, 32)
    X = np.array([
        green,
        black,
        white,
        red,
        green,
        blue,
        green,
        red,
        white,
        black,
    ])
    # nn expects batch x chan x width x height
    X = np.rollaxis(X, 3, 1)

    # Again, no assertion. Need to manually inspect the result.
    _, fname = tempfile.mkstemp(suffix='.mp4')
    ani = vizlib.animations.from_neural_network(nn, X)
    ani.save(fname, fps=0.5, extra_args=['-vcodec', 'libx264'])
    subprocess.call(['gnome-open', fname])
