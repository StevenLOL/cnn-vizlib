#!/usr/bin/env python2.7
# encoding: utf-8
'''
'''
import matplotlib.pyplot as plt
import os

path = os.path.split(__file__)[0]
if path == '': path = '.'
files = sorted([
    os.path.join(path, f) for f in os.listdir(path) if f.endswith('.png')
], key=lambda x: (int(x.split('_')[1]), int(x.split('_')[2][:-len('0001.png')])))


data = [
    ((0,    315),  (0,    360),  3),
    ((0,    45),   (0,    90),   2),
    ((0,    135),  (0,    180),  4),
    ((0,    225),  (0,    270),  5),

    ((40,   315),  (40,   360),  3),
    ((40,   55),   (40,   90),   1),
    ((40,   135),  (40,   180),  4),
    ((40,   235),  (40,   270),  6),

    ((135,  315),  (135,  360),  3),
    ((135,  55),   (135,  90),   1),
    ((135,  135),  (135,  180),  4),
    ((135,  235),  (135,  270),  2),

    ((225,  315),  (225,  360),  3),
    ((225,  55),   (225,  90),   5),
    ((225,  135),  (225,  180),  4),
    ((225,  235),  (225,  270),  1),

    ((315,  315),  (315,  360),  3),
    ((315,  55),   (315,  90),   6),
    ((315,  135),  (315,  180),  4),
    ((315,  235),  (315,  270),  5),
]

for (s1, e1), (s2, e2), c in data:
    s1 %= 360
    e1 %= 360
    s2 %= 360
    e2 %= 360

    f, axs = plt.subplots(nrows=1, ncols=2)
    f1 = os.path.join(path, 'die_{}_{}0001.png'.format(s1, e1))
    axs[0].imshow(plt.imread(f1))
    axs[0].set_title(f1)

    f2 = os.path.join(path, 'die_{}_{}0001.png'.format(s2, e2))
    axs[1].imshow(plt.imread(f2))
    axs[1].set_title(f2)

    f.canvas.draw()
    plt.waitforbuttonpress()
