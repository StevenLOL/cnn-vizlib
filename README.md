# cnn-vizlib

Implement multiple convolution network visualization methods on top of [theano](https://github.com/Theano/Theano) and [lasagne](https://github.com/Lasagne/Lasagne).

# Romulus

    local$      ssh -L 8080:localhost:9999 romulus
    romulus$    jupyter notebook

Romulus is now available via a notebook at `localhost:8080` on your local
machine.

# Methods

- activation maximization
- saliency maps
    - occlusion based
    - gradient based
- deconvolution network

# Dependencies

- theano
- lasagne
- numpy

# Data

The die dataset was generated using
[blender](https://www.blender.org/download/).

To regenerate first download blender, then:

    python2 blender.py <blender_path>

This will generate a bunch of different rotations of the die, and place their
class name in the filename.

To generate the `.npy` files needed for vizlib use:

    python2 compress.py

This should generate `./vizlib/data/die.{X,y}.npy'. These should also be stored
in the repository for convenience.

If at some point you need to change the resolution, you should edit the
`die.blend` file in the `die` folder.
