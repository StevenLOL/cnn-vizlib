#!/usr/bin/env python2.7
# encoding: utf-8
'''
'''
import numpy as np
import scipy.io

xfname = 'die.X.npy'
yfname = 'die.y.npy'
X = np.load(xfname)
y = np.load(yfname)
scipy.io.savemat('die.mat', dict(X=X, y=y))
