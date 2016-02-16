# cnn-vizlib

Implement multiple convolution network visualization methods on top of [theano](https://github.com/Theano/Theano) and [lasagne](https://github.com/Lasagne/Lasagne).

# Romulus

    local$      ssh -L 8080:localhost:9999 romulus
    romulus$    jupyter notebook

Romulus is now available via a notebook at `localhost:8080` on your local
machine.

Romulus is running:
- `CentOS release 6.7 (Final)` as echoed by `cat /etc/redhat-release`.

Also installed is the `anaconda` package which is a collection of useful python
packages. The package manager that comes with anaconda is `conda`. There seem to
be no conflicts between `conda` and `pip` installations.

This `conda` installation is installed in directory where the `wojtek` account
cannot write, so instead it should be cloned locally. I used the following
command:

    conda create -n conda --clone=/zfsdata/conda/conda

This created a new environment at `/home/wojtek/.conda/envs/conda` which can be
activated using:

    source activate conda

I could now install `opencv` locally:

    conda install opencv

# Working on cnn-vizlib

To install a python package as editable use `pip -e`. So to install `vizlib` use:

    pip install -e cnn-vizlib --user

The `--user` installs the package for the current user only, which is ideal on
remote desktops where you do not have root access.

# Packages

`<TODO: FIND A BETTER SOLUTION>`

Currently `cnn-vizlib` requires bleeding-edge installation of `lasagne` (which
in turn requires `theano`). Both can be installed using:

    pip install --upgrade https://github.com/Theano/Theano/archive/master.zip --user
    pip install --upgrade https://github.com/Lasagne/Lasagne/archive/master.zip --user
    
    In [3]: lasagne.__version__
    Out[3]: '0.2.dev1'

    In [5]: theano.__version__
    Out[5]: '0.7.0.dev-15c90dd36c95ea2989a2c0085399147d15a6a477'

# Tests

You can run the tests from the `cnn-vizlib` by using `py.test` command. If the
`py.test` command is not available install it using `pip install pytest --user`.

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

# What needs to be done

Networks need to be trained. Each network is trained by:

- Loading the dataset
- Setting up a network
- Training the network

There might be multiple networks to consider. In this case it is already
beneficial to move to Romulus for speedups.

In addition to training the network, the network need to be inspected. The
Deconvolutional method can not be applied to every type of network.

Network inspection should be separated from network training so the trained
models should be serialized. Lasagne networks are easily serialized. The topmost
layer holds a reference to the layers below and so on. This means the structure
can easily be serialized using pickle:

    with open('mypickle.pickle, 'wb') as fh:
        pickle.dump(topmost_layer, fh, -1)
    topmost_layer_loaded = pickle.load(open('mypickle.pickle', 'rb'))

I had less success serializing a `nolearn.lasagne.NeuralNet` since you might
define progress functions during training and such, which then need to be
available during load time. This leads to complications when reusing a network
across different scripts, since the local helper functions won't be available.

# Restrictions of Deconvolutional network

- The output shape of the Convolutional layer should be the same as the input shape.
  This requires a special padding setting called `pad='same`'. For a filter of size
  `F = 2n + 1` (where `F` necessarily is uneven, since `n` is an integer), an
  output of size `(N-F+1)x(M-F+1)` is produced when no padding is used.
  Zero-padding both sides of the output with `n` zeros leads to `NxM` output.
- The only supported nonlinearity is `Rectify`.
- The only supported form of pooling is MaxPooling in two dimensions, and this
  should be done using pooling with a `stride` equal to the `pool_size`, and a
  `pool_size` that is a factor of the input size of said `MaxPool2D` layer.

These restrictions are easy to adhere to if you ensur the following (as is
common in the literature):

- An input image size that is even
- Uneven filters
- `2x2` maxpooling
