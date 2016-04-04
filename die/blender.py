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

References:
    http://blender.stackexchange.com/questions/6817/how-to-pass-command-line-arguments-to-a-blender-python-script
'''
import subprocess
import sys

blender_exec_path = sys.argv[1]

for a in range(0, 360):
    for b in range(0, 360):
        output = 'die_{a}_{b}'.format(**locals())
        cmd = '{blender_exec_path} -b die.blend -P rotator.py -o {output} -F jpeg -x 1 -f 1 -- -a {a} -b {b}'.format(**locals())
        retcode = subprocess.call(cmd, shell=True)
        print(cmd, retcode)
