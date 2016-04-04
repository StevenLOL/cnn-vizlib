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
a = deg2rad(float(args[args.index('-a') + 1]))
b = deg2rad(float(args[args.index('-b') + 1]))

for name, dist in (('Camera', 3), ('Lamp', 6)):
    obj = bpy.data.objects[name]
    obj.location.x = math.cos(a) * math.sin(b) * dist
    obj.location.y = math.sin(a) * math.sin(b) * dist
    obj.location.z = math.cos(b) * dist
    obj.rotation_euler.y = b
    obj.rotation_euler.z = a
