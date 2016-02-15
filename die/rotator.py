#!/usr/bin/env python2.7
# encoding: utf-8
'''
'''
import bpy
import sys
import math

def deg2rad(degrees):
    return degrees / 180.0 * math.pi

args = sys.argv[sys.argv.index('--') + 1:]
x = deg2rad(float(args[args.index('-x') + 1]))
y = deg2rad(float(args[args.index('-y') + 1]))
z = deg2rad(float(args[args.index('-z') + 1]))

dice = bpy.data.objects['Cube']
dice.rotation_euler.x = x
dice.rotation_euler.y = y
dice.rotation_euler.z = z

# rendering done through commandline options.

