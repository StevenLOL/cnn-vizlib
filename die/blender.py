#!/usr/bin/env python2.7
# encoding: utf-8
'''I took a simple dice into blender and wanted to render it under different
rotations.

Doing this manually would be very time consuming
(especially if I would forget something, and I am sure I will)
so I tried to automated the process.

Usage:
    # run blender on file <file>.blend
    ./blender -b <file>.blend \
            # run the script called blender.py
            -P blender.py \
            # render to <output>
            -o <output>
            # use the file format jpeg
            -F jpeg
            # append the extension to <output>
            -x 1
            # render frame 1
            -f 1
            --
            # all arguments are available to blender.py
            # blender itself will ignore all arguments after --
            # so we can use these to pass arguments to our script.
            # the rotation of our cube:
            -x 0
            -y 0
            -z 45

I determined the following by rotating manually through blender:

    x   z     face
    0   45    6
        135   5
        225   1
        315   2
    90  45    3
        135   5
        225   4
        315   2
    180 45    1
        135   5
        225   6
        315   2
    270  45   4
        135   5
        225   3
        315   2

Frequency of faces in the above schema:

    5    4
    2    4
    6    2
    4    2
    3    2
    1    2

Thus we can leave out two faces containing 2.
I know for a fact that w/ `x = 0` and `x = 180` the faces for 2 is changed.
So let's use those two.

The `z` axis is responsible for ambiguity. For offset values from 45:

    0   full face
    20  ok
    40  difficult
    45  ambiguous
    45+ irrelevant

I.e., `z` = 45 will give a full face. Hence the term 'offset' value.

For `y` I found that the value can be varied between -70 and 10.

References:
    http://blender.stackexchange.com/questions/6817/how-to-pass-command-line-arguments-to-a-blender-python-script
'''
import subprocess
import numpy as np
import sys

blender_exec_path = sys.argv[1]

rotation_data = [
    #x, z, face
    [0  , 45 ,   6],
    [0  , 135,   5],
    [0  , 225,   1],
    [0  , 315,   2],
    [90 , 45 ,   3],
    [90 , 135,   5],
    [90 , 225,   4],
    [180, 45 ,   1],
    [180, 225,   6],
    [180, 315,   2],
    [270,  45,   4],
    [270, 225,   3],
]

n_z_values = 5
n_y_values = 5

z_min = 0
z_max = 40

# -70 is really pushing it.
y_min = -50
y_max = 10

z_values = np.linspace(z_min, z_max, n_z_values).round()
y_values = np.linspace(y_min, y_max, n_y_values).round()

for x, z0, face in rotation_data:
    for z_offset in z_values:
        z = z0 + z_offset
        for y in y_values:
            output = 'die_{x}_{y}_{z}__{face}'.format(**locals())
            cmd = '{blender_exec_path} -b die.blend -P rotator.py -o {output} -F jpeg -x 1 -f 1 -- -x {x} -y {y} -z {z}'.format(**locals())
            retcode = subprocess.call(cmd, shell=True)
            print(cmd, retcode)
