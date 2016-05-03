#!/usr/bin/env python2.7
# encoding: utf-8
'''Simple wrapper around ImageMagick
'''
import subprocess
import os
import numpy as np
import tempfile
import matplotlib.image as mpimg


def from_numpy_arrays(numpy_arrays, output_file, **kwargs):
    fnames = []

    # save all arrays to a file of their own
    for arr in numpy_arrays:
        _, fname = tempfile.mkstemp(suffix='.png')
        fnames.append(fname)
        mpimg.imsave(fname, arr)

    # convert [input-option] input-file [output-option] output-file
    # I only tested the -delay argument, which had to be placed like
    # 'input-option'. If something does not work, consider splitting it
    # into output option.
    input_options = sum([['-{}'.format(k), str(v)] for k, v in kwargs.iteritems()], [])
    output_options = []
    rc = subprocess.call(['convert', ] + input_options + fnames + output_options + [output_file])
    assert rc == 0

    for fname in fnames:
        os.remove(fname)

    return output_file

def from_activations(activations, output_file, **kwargs):
    # Activations are of the form: nbatch x nchannels x width x height
    # We wish to display these as grayscale images, with the channels side by side.
    horizontally_concatenated_activations = [np.hstack(row) for row in activations]
    return from_numpy_arrays(horizontally_concatenated_activations, output_file, **kwargs)
