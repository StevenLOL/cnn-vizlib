#!/usr/bin/env python2.7
# encoding: utf-8
'''
'''
import sys
import math

def deg2rad(degrees):
    return degrees / 180.0 * math.pi

a = deg2rad(float(sys.argv[1]))
b = deg2rad(float(sys.argv[2]))

print('{:.2f} {:.2f} {:.2f}'.format(math.cos(a) * math.sin(b) * 3, math.sin(a) * math.sin(b) * 3, math.cos(b) * 3))
