#!/usr/bin/env python2.7
# encoding: utf-8
'''Use the `convert` command to create gifs for each class.
'''
from subprocess import call
import os
import pickle

def confirm(dialogue):
    dialogue = dialogue + ' (Y/n)'
    inp = None
    while inp not in ('', 'y', 'n'):
        inp = raw_input(dialogue).lower()
    return inp != 'n'

path = os.path.join(os.path.split(__file__)[0] or '.', 'imgs')

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

sources = {}
count = 0
for i in range(len(first_rot_axis) - 1):
    for j in range(len(second_rot_axis)):
        ra1 = range(first_rot_axis[i], first_rot_axis[i+1])
        ra2 = range(second_rot_axis[j][0], second_rot_axis[j][1] + 1)
        c = classifications[i][j]
        if c is None:
            continue
        fnames = []
        for a1 in ra1:
            for a2 in ra2:
                a2 = a2 % 360
                fnames.append(os.path.join(path, 'die_{a1}_{a2}0001.png'.format(**locals())))
        names = ' '.join(fnames)
        output = 'gifs/{count}.gif'.format(**locals())
        cmd = 'convert {names} {output}'.format(**locals())
        #if confirm('Executing: {}'.format(cmd)):
        call(cmd, shell=True)
        count += 1
        sources[output] = fnames

with open(os.path.join('gifs', 'sources.p'), 'wb') as fh:
    pickle.dump(sources, fh, -1)
