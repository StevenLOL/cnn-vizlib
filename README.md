# CNN-VIZLIB

Library to visualize features learned by Convolutional Neural Networks. Built on
top of [theano](https://github.com/Theano/Theano) and [lasagne](https://github.com/Lasagne/Lasagne).

# Install

    $ pip install -r requirements.txt
    $ pip install -e .

OpenCV can not be installed through pip. 

# Methods

Activation Maximization (`activation_maximization/`)
: Visualize a filter by finding the input that maximizes its activation.

Deconvolutional Networks (`deconvolution/`)
: Visualize a filter with respect to a certain input image by projecting the
filter's activation on a given image back to input space.

Gradient Based Method (`class_saliency_map/`)
: Assign saliency scores to individual pixels with respect to a certain filter
by assuming the activation can be approximated as a simple linear function.

Occlusion Method (`class_saliency_map/`)
: Assign saliency scores to individual pixels with respect to a certain filter
by occluding a region centered on an individual pixel and observing the effect
on the networks output.

# Examples

TODO

## Virtual Environments

Virtual environments are great when working on a machine that is not owned by
you. The following steps show how to setup a virtual environment on a machine on
which you do not have root access. All files are installed to `$HOME/local`.

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

# Working Remotely

A great method for working interactively on headless servers is the IPython
notebook (today called Jupyter notebook). The following shows how to setup the
notebook on the remote side, and how to connect to them locally using SSH
tunnels.

## Simple Setup

A simple setup, with a single remote machine and a single local machine.

This assumes you are using the old iPython notebook. See below for a Jupyter
notebook example.

    remote$    ipython profile create notebookserver
    # will generate a notebook config file
    remote$     vim ~/.ipython/profile_notebookserver/ipython_notebook_config.py

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

If the information you are viewing is private, consider using https and
passwords.

## Proxy Setup

This setup looks as follows:

    local
    head [remote node that is accessible from local, but not for computing]
    compute [compute node that is only accessible from the head node]

We create a tunnel such that localhost:8889 arrives at compute:9999, through
head:9998:

    local$ 	ssh -L 8889:localhost:9998 head ssh -L 9998:localhost:9999 -N compute
    head$ 	jupyter notebook --generate-config
                vim ~/.jupyter/jupyter_notebook_config.py
                # enter same configuration as above
                ssh compute
    compute$    jupyter notebook

Connect with local browser to `localhost:8889`

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
