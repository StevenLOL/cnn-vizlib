#!/usr/bin/env python2.7
# encoding: utf-8
'''Inspected the die images manually. Noted what rotation corresponds to what class.
Here we generate the dataset.
'''
import os
import matplotlib.pyplot as plt
import numpy as np


path = os.path.split(__file__)[0]
if path == '': path = '.'
files = sorted([
    os.path.join(path, f) for f in os.listdir(path) if f.endswith('.png')
], key=lambda x: (int(x.split('_')[1]), int(x.split('_')[2][:-len('0001.png')])))

start_idx = None
current_idx = 0
class_ = None
labels = [None for f in files]

def plot():
    global current_idx, ax, fig
    x = plt.imread(files[current_idx])
    ax.imshow(x)
    ax.set_title('{} {}'.format(files[current_idx], class_))
    fig.canvas.draw()

def press(event):
    global current_idx, start_idx, class_, ax, fig

    k = event.key

    if current_idx == len(files):
        import IPython; IPython.embed()

    if k == 'left':
        current_idx -= 5
    elif k == 'down':
        current_idx -= 15
    elif k == 'shift+left':
        current_idx -= 360 * 5
    elif k == 'shift+down':
        current_idx -= 360 * 15

    elif k == 'right':
        current_idx += 5
    elif k == 'up':
        current_idx += 15
    elif k == 'shift+right':
        current_idx += 360 * 5
    elif k == 'shift+up':
        current_idx += 360 * 15

    elif k == ' ':
        if start_idx is not None:
            assert class_ is not None, 'je suis retarded'
            labels[start_idx:current_idx] = [class_] * len(labels[start_idx:current_idx])
            class_ = start_idx = None
    else:
        class_ = int(event.key)
        start_idx = current_idx

    plot()

fig, ax = plt.subplots()
fig.canvas.mpl_connect('key_press_event', press)
plot()
plt.show()
