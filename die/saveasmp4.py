#!/usr/bin/env python2.7
# encoding: utf-8
'''
'''
import numpy as np
import os
import sys
import pickle
import vizlib

# Needed to load the network pickle
sys.path.insert(0, './../die_experiments/')
import netshared
assert netshared # silence pyflakes

die_dir = './'
mp4_dir = './mp4s/'

def ls(dir):
    return [os.path.join(dir, f) for f in os.listdir(dir)]

def load_dataset():
    X = np.load(os.path.join(die_dir, 'die.32.X.npy'))
    y = np.load(os.path.join(die_dir, 'die.y.npy'))
    ds = vizlib.data.DataSet(X, y).standardize('global')
    return ds

def load_mask():
    first_rot_axis = (0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330, 360)
    second_rot_axis = ((50, 90), (140, 180), (230, 270), (320, 360))
    classifications = [
        [2,     4,    5,     3],
        [2,     4,    None,  3],
        [None,  4,    6,     3],
        [1,     4,    6,     3],
        [1,     4,    None,  3],
        [None,  4,    2,     3],
        [5,     4,    2,     3],
        [5,     4,    None,  3],
        [None,  4,    1,     3],
        [6,     4,    1,     3],
        [6,     4,    None,  3],
        [None,  4,    5,     3],
    ]
    count = 0
    mask = []
    for i in range(len(first_rot_axis) - 1):
        for j in range(len(second_rot_axis)):
            ra1 = range(first_rot_axis[i], first_rot_axis[i+1])
            ra2 = range(second_rot_axis[j][0], second_rot_axis[j][1] + 1)
            c = classifications[i][j]
            if c is None:
                continue
            mask.extend([count] * len(ra1) * len(ra2))
            count += 1
    mask = np.array(mask)
    return mask

def load_nn(experiment_folder):
    # Try to load the best neuralnetwork for each experiment,
    # where best is defined as having the highest final validation_score
    nn_files = ls(experiment_folder)
    nns = [pickle.load(open(f, 'rb')) for f in nn_files]
    best_nn = max(nns, key=lambda x: x.train_history_[-1]['valid_accuracy'])
    return best_nn

# Simply load the network, then for each mask create a movie
# Name it after the mask, but also put in the class,
# so that way we can possibly merge them later.

network_folder = sys.argv[1]
nn = load_nn(network_folder)
ds = load_dataset()
mask = load_mask()

if network_folder.endswith('/'):
    network_folder = network_folder[:-1]
mp4_dir = os.path.join(mp4_dir, os.path.split(network_folder)[1])
if not os.path.exists(mp4_dir):
    os.makedirs(mp4_dir)
print('Saving to', mp4_dir)

width = len(str(mask.max()))
for m in range(mask.max() + 1):
    X = ds.X[mask == m]
    y = ds.y[mask == m][0]

    # I noticed that the animation is very non-smooth,
    # since the animation tends to jump whenever the second axis is rotated.
    # the second axis contains 41 degrees always I think.
    # so we need to reverse every other 41 images.
    reverse = False
    i = 0
    ranges = []
    while i < len(X):
        i_start = i
        i_end = i_start + 41
        r = np.arange(i_start, i_end)
        if reverse:
            r = r[::-1]
        reverse = not reverse
        ranges.extend(r)
        i = i_end
    X = X[ranges]

    output_file = os.path.join(mp4_dir, '{:0>{width}}_{}.mp4'.format(m, y, width=width))
    ani = vizlib.animations.from_neural_network(nn, X)
    ani.save(output_file, fps=30, extra_args=['-vcodec', 'libx264'])

    print("DONE WITH", '{:0>{width}}_{}.mp4'.format(m, y, width=width))
