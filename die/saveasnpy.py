#!/usr/bin/env python2.7
# encoding: utf-8
'''
'''
import matplotlib.pyplot as plt
from skimage import color
import numpy as np
import pandas as pd

import os

path = os.path.join(os.path.split(__file__)[0] or '.', 'imgs')

# In some cases a side would be invalid
# (it would be at the bottom, or at the left)
# This occurs under different conditions.
# Anyway, you should read the following table as follows:
# first_rot_axis is a range at which the rotation is valid.
# (second_rot_axis[j], second_rotation_axis[j+1]) forms the second_rotation_axis
# range over which the classification is valid.
# If the classification[i][j] is invalid, the classifications[i][j] is None.
first_rot_axis = ((50, 90), (140, 180), (230, 270), (320, 360))
second_rot_axis = (0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330, 360)
classifications = zip(*[
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
])

ys = []
xs = []
for cs, (f, t) in zip(classifications, first_rot_axis):
    for j in range(len(second_rot_axis) - 1):
        ra1 = range(f, t + 1)
        ra2 = range(second_rot_axis[0], second_rot_axis[1])
        c = cs[j]
        if c is None:
            continue
        for a1 in ra1:
            a1 = a1 % 360
            for a2 in ra2:
                fname = os.path.join(path, 'die_{a1}_{a2}0001.png'.format(**locals()))
                xs.append(color.rgb2gray(plt.imread(fname)))
                ys.append(c)

xfname = 'die.X.npy'
yfname = 'die.y.npy'
np.save(xfname, np.array(xs))
np.save(yfname, np.array(ys))

print('Saved to {xfname} and {yfname}'.format(**locals()))
print(pd.value_counts(ys))
