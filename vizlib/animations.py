#!/usr/bin/env python2.7
# encoding: utf-8
'''
'''
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import theano
import lasagne
import matplotlib.gridspec as gridspec
from itertools import izip

def from_numpy_arrays(numpy_arrays, output_file, dpi=100, fps=30):
    f, ax = plt.subplots(1)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    im = ax.imshow(numpy_arrays[0],cmap='gray',interpolation='none')

    def update_img(idx):
        im.set_data(numpy_arrays[idx % len(numpy_arrays)])
        return im,

    # When too little frames are provided, the animation cuts short.
    # I rather have a cyclical animation, than missing part of the animation...
    # So that is what this does.
    n_frames = max(len(numpy_arrays), 30)
    ani = animation.FuncAnimation(f, update_img, n_frames, blit=True)
    writer = animation.writers['ffmpeg'](fps=fps)
    ani.save(output_file, writer=writer, dpi=dpi)

    return ani

def from_activations(activations, output_file, **kwargs):
    # Activations are of the form: nbatch x nchannels x width x height
    # We wish to display these as grayscale images, with the channels side by side.
    horizontally_concatenated_activations = [np.hstack(row) for row in activations]
    return from_numpy_arrays(horizontally_concatenated_activations, output_file, **kwargs)

def ensure_plottable_shape(x):
    if len(x.shape) == 1:
        return x.reshape((1, ) + x.shape)
    return x

def cwh_to_whc(x):
    # Strictly speaking this is not whc, since c is left off when it is 1.
    return np.rollaxis(x, 0, 3).squeeze()

def align_horizontally(X):
    if len(X.shape) < 3:
        X = X.reshape(X.shape + (1, ) * (3 - len(X.shape)))
    assert len(X.shape) == 3, 'did not think about any other scenario'
    # For each x, we would like there to be a padding strip in between.
    # Almost like a join.
    padding_width = 1
    padding = np.ones((X.shape[1], padding_width), dtype=X.dtype)
    joined = []
    for x in X:
        joined.append(x)
        joined.append(padding)
    joined.pop() # remove the last erronous padding
    return np.hstack(joined)

def from_neural_network(neural_network, X):
    layers = neural_network.get_all_layers()
    input_layer = layers.pop(0)
    f = theano.function(
        [input_layer.input_var],
        lasagne.layers.get_output(layers),
    )
    activations_per_layer = f(X)
    activations_per_layer.insert(0, X)
    # XXX: assumed all activations fit in memory. This might not be true
    # for larger networks, while we still want to have a single animation.

    f = plt.figure()
    shapes = [activations.shape[1:] for activations in activations_per_layer]
    nn_plotter = NeuralNetworkPlotter(shapes)

    def init_func():
        return nn_plotter.init_func([activations[0] for activations in activations_per_layer])

    def update(idx):
        idx = idx % len(X)
        return nn_plotter.update([activations[idx] for activations in activations_per_layer])

    # When too little frames are provided, the animation cuts short.
    # I rather have a cyclical animation, than missing part of the animation...
    # So that is what this does.
    n_frames = max(len(X), 30)
    ani = animation.FuncAnimation(
        f,
        update,
        n_frames,
        init_func=init_func,
        blit=True
    )
    # For whatever reason the figure needs to be closed...
    # Otherwise the animation will not always play.
    plt.close()
    return ani

class NeuralNetworkPlotter(object):

    def __init__(self, input_shapes):
        self.plotters = []
        for idx, input_shape in enumerate(input_shapes):
            plotter = self.create_plotter(input_shape, idx == 0, idx + 1 == len(input_shapes))
            self.plotters.append(plotter)

    def init_func(self, inputs):
        n_rows = len(self.plotters)
        n_columns = max(p.n_required_axes for p in self.plotters)
        gs = gridspec.GridSpec(n_rows, n_columns)

        artists = []
        for i, (p, x) in enumerate(izip(self.plotters, inputs)):
            axes = []
            if p.n_required_axes == 1:
                axes.append(plt.subplot(gs[i, :]))
            else:
                for j in xrange(p.n_required_axes):
                    axes.append(plt.subplot(gs[i, j]))
            for ax in axes:
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['left'].set_visible(False)
                ax.spines['bottom'].set_visible(False)
            artist = p.init_func(x, axes)
            artists.extend(artist)

        return artists

    def update(self, inputs):
        updated_artists = []
        for p, x in izip(self.plotters, inputs):
            updated_artist = p.update(x)
            updated_artists.extend(updated_artist)
        return updated_artists

    def create_plotter(self, input_shape, is_first, is_last):
        sl = len(input_shape)

        if sl == 3 and is_first:
            return InputLayerPlotter(input_shape)
        if sl == 3:
            return ConvLayerPlotter(input_shape)
        if sl == 1 and is_last:
            return PosteriorPlotter(input_shape)
        if sl == 1:
            return DenseLayerPlotter(input_shape)

        raise ValueError('Unsupported input shape {}'.format(input_shape))


class DenseLayerPlotter(object):

    def __init__(self, input_shape):
        self.cmap = 'gray'
        self.ax = None
        self.artists = []

    def transform(self, x):
        return x[None, :] # Broadcast to (y, 1) from (y, )

    def init_func(self, x, axes):
        self.ax = axes[0]
        self.artists.append(self.ax.imshow(self.transform(x), cmap=self.cmap, interpolation='none'))
        self.ax.get_xaxis().set_visible(False)
        self.ax.get_yaxis().set_visible(False)
        return self.artists

    def update(self, x):
        '''Returns the updated artists'''
        self.artists[0].set_data(self.transform(x))
        return self.artists

    @property
    def n_required_axes(self):
        return 1

class InputLayerPlotter(DenseLayerPlotter):

    def __init__(self, input_shape):
        super(InputLayerPlotter, self).__init__(input_shape)

    def transform(self, x):
        return np.rollaxis(x, 0, 3).squeeze()

class ConvLayerPlotter(object):

    def __init__(self, input_shape):
        self.axes = None
        self.artists = []
        self.n_required_axes = input_shape[0]

    def init_func(self, xs, axes):
        self.axes = axes
        # Create subaxes for each channel to be plotted on
        for ax, x in izip(axes,xs):
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            artist = ax.imshow(x, cmap='gray', interpolation='none')
            self.artists.append(artist)
        return self.artists

    def update(self, xs):
        for artist, x in izip(self.artists, xs):
            artist.set_data(x)
        return self.artists

class PosteriorPlotter(object):

    def __init__(self, input_shape):
        self.ax = None
        self.artists = None     # The container of the rectangles of the bars
        self.text_artists = [] # The text showing on top of the bars
        self.width = 0.35

    def init_func(self, x, axes):
        ax = self.ax = axes[0]
        ind = np.arange(len(x))
        rects = self.artists = list(ax.bar(ind, x, self.width))

        ax.set_ylim([0.0, 1.0 + 0.2])
        ax.set_yticks([0, 0.5, 1.0], minor=False)
        ax.get_yaxis().set_visible(False)

        # Ensure some spacing around the boxes
        ax.set_xlim(-1, len(x) + 1)
        ax.set_xticks(ind + self.width/2., minor=False)
        ax.set_xticklabels(ind, minor=False)
        ax.xaxis.set_ticks_position('bottom')

        # Draw the posterior values on top
        for rect in rects:
            height = rect.get_height()
            x = rect.get_x() + rect.get_width() / 2.
            y = 1.05 * height
            text_artist = ax.text(x, y, '%.2f' % height, ha='center', va='bottom')
            self.text_artists.append(text_artist)

        ax.set_ylabel('posterior')
        return self.artists + self.text_artists

    def update(self, x):
        for v, rect, text in izip(x, self.artists, self.text_artists):
            rect.set_height(v)
            text.set_text('%.2f' % v)
            text.set_y(v * 1.05)
        return self.artists + self.text_artists

    @property
    def n_required_axes(self):
        return 1
