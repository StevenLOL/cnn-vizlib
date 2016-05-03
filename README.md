# cnn-vizlib

Implement multiple convolution network visualization methods on top of [theano](https://github.com/Theano/Theano) and [lasagne](https://github.com/Lasagne/Lasagne).

# Installation

Using [virtualenv](http://docs.python-guide.org/en/latest/dev/virtualenvs/):

    $ pip install virtualenv
    $ pip install virtualenvwrapper
    $ source virtualenvwrapper.sh
    $ mkvirtualenv venv
    $ workon venv
    $ cd cnn-vizlib/
    $ pip install -r requirements.txt
    $ pip install -e .

To install python (required for the shared object files):

    deactivate
    wget https://www.python.org/ftp/python/2.7.9/Python-2.7.9.tgz
    tar xzvf Python-2.7.9.tgz 
    cd Python-2.7.9
    ./configure --prefix=$HOME/.local --enable-shared
    make -j7 && make install

To install [opencv](http://opencv.org/downloads.html):

    $ wget https://github.com/Itseez/opencv/archive/2.4.12.zip
    $ unzip 2.4.12.zip
    $ cd opencv-2.4.12
    $ mkdir release
    $ cd release
    $ cmake -D PYTHON_EXECUTABLE=`which python`\
        -D PYTHON_LIBRARY=$HOME/.local/lib/libpython2.7.so\
        -D CMAKE_BUILD_TYPE=RELEASE\
        -D CMAKE_INSTALL_PREFIX=$HOME/.local ..
    $ make && make install

    $ ln -s $HOME/.local/lib/python2.7/site-packages/cv.py \
        $HOME/.virtualenvs/venv/lib/python2.7/site-packages/cv.py
    $ ln -s $HOME/.local/lib/python2.7/site-packages/cv2.so \
        $HOME/.virtualenvs/venv/lib/python2.7/site-packages/cv2.so

Verify that everything is correct:

    $ cd cnn-vizlib
    $ py.test

# Notebooks

All liacs servers that have GPU are not accessible from outside networks.
However, we can run a notebook server on them and access it through ssh
tunneling.

    remote$    ipython profile create notebookserver
    # will generate a notebook config file
    remote$    vim ~/.ipython/profile_notebookserver/ipython_notebook_config.py

    # insert following lines
    c = get_config()
    c.NotebookApp.ip = 'localhost'
    c.NotebookApp.port = 9999
    c.NotebookApp.open_browser = False
    # save and exit

    remote$    ipython notebook --profile=notebookserver

Now forward the port

    local$     ssh -L 8889:localhost:9999 remote

Now point your browser to the following url:

    http://localhost:8889

You can also use `https` and passwords, but I did not bother as the information
is not private.

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

# Das

The problem with the das is that root nodes do not have access to a GPU.
For this reason we need to run the notebook on a child node, but we can only
connect to this node through the root node. So we need to chain port forwarding,
and configure jupyter notebook. Here are the commands that I ran:

    local$ 	ssh -L 8889:localhost:9999 dasvu ssh -L 9999:localhost:9999 -N node009
    dasvu$ 	jupyter notebook --generate-config
                vim ~/.jupyter/jupyter_notebook_config.py
                # enter same configuration as above
                ssh node009
    node009$    jupyter notebook

Connect with local browser to `localhost:8889`

# Methods

- activation maximization
- saliency maps
    - occlusion based
    - gradient based
- deconvolution network

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
