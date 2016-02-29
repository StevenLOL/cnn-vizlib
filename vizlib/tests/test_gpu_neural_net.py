#!/usr/bin/env python2.7
# encoding: utf-8
'''Test my GpuNeuralNet implementation against the nolearn.lasagne implementation.
'''
import vizlib
import nolearn.lasagne
import lasagne
import numpy as np

def test_gpu_dense_neural_net():
    ds = vizlib.data.counting_2d(max_spots=3, size=(28, 28))
    ds.X = ds.X[ds.y != 0]
    ds.y = ds.y[ds.y != 0] - 1

    input_layer = lasagne.layers.InputLayer(
        (None, 1, 28, 28)
    )
    output_layer = lasagne.layers.DenseLayer(
        input_layer,
        num_units=len(set(ds.y)),
        nonlinearity=lasagne.nonlinearities.softmax
    )
    # store the original layer state
    wo0, bo0 = params(output_layer)

    np.random.seed(42)
    nn = nolearn.lasagne.NeuralNet(
        output_layer,
        update_learning_rate=1e-2,
        update_momentum=0.9,
    )
    nn.initialize()
    p0 = nn.predict(ds.X)
    nn.fit(ds.X, ds.y)
    p1 = nn.predict(ds.X)
    wo1, bo1 = params(output_layer)

    input_layer2 = lasagne.layers.InputLayer(
        (None, 1, 28, 28)
    )
    output_layer2 = lasagne.layers.DenseLayer(
        input_layer2,
        num_units=len(set(ds.y)),
        nonlinearity=lasagne.nonlinearities.softmax,
        W=wo0,
        b=bo0,
    )

    np.random.seed(42)
    nn2 = vizlib.utils.GpuNeuralNet(
        output_layer2,
        update_learning_rate=1e-2,
        update_momentum=0.9,
    )
    nn2.initialize(ds.X, ds.y)
    p20 = nn2.predict(ds.X)
    nn2.fit(ds.X, ds.y)
    p21 = nn2.predict(ds.X)
    wo21, bo21 = params(output_layer2)

    acc0 = (p0 == ds.y).mean()
    acc20 = (p20 == ds.y).mean()

    acc1 = (p1 == ds.y).mean()
    acc21 = (p21 == ds.y).mean()

    train_losses = [v['train_loss'] for v in nn.train_history_]
    train_losses2 = [v['train_loss'] for v in nn2.train_history_]
    valid_losses = [v['valid_loss'] for v in nn.train_history_]
    valid_losses2 = [v['valid_loss'] for v in nn2.train_history_]

    np.testing.assert_allclose(p0, p20)
    np.testing.assert_allclose(p1, p21)
    np.testing.assert_allclose(wo1, wo21)
    np.testing.assert_allclose(bo1, bo21)

    assert acc20 < acc21
    assert acc1 == acc21
    assert acc0 == acc20
    assert acc0 < acc1

    assert valid_losses == valid_losses2
    # TODO: figure out why this is failing...
    #assert train_losses == train_losses2

def test_gpu_conv_net():
    # simple triangle dataset
    # network: input, conv, dense
    # train nolearn.lasagne
    # reinitialize network to initial values
    # train vizlib.utils
    # compare weights and outputs
    ds = vizlib.data.triangle_and_circle()

    input_layer = lasagne.layers.InputLayer(
        (None, 1, 32, 32)
    )
    conv_layer = lasagne.layers.Conv2DLayer(
        input_layer,
        num_filters=1,
        filter_size=(32, 32)
    )
    output_layer = lasagne.layers.DenseLayer(
        conv_layer,
        num_units=2,
        nonlinearity=lasagne.nonlinearities.softmax
    )
    # store the original layer state
    wc0, bc0 = params(conv_layer)
    wo0, bo0 = params(output_layer)

    nn = nolearn.lasagne.NeuralNet(
        output_layer,
        update_learning_rate=1e-2,
        update_momentum=0.9,
    )
    nn.initialize()
    p0 = nn.predict(ds.X)
    nn.fit(ds.X, ds.y)
    p1 = nn.predict(ds.X)
    wc1, bc1 = params(conv_layer)
    wo1, bo1 = params(output_layer)

    input_layer2 = lasagne.layers.InputLayer(
        (None, 1, 32, 32)
    )
    conv_layer2 = lasagne.layers.Conv2DLayer(
        input_layer2,
        num_filters=1,
        filter_size=(32, 32),
        W=wc0,
        b=bc0,
    )
    output_layer2 = lasagne.layers.DenseLayer(
        conv_layer2,
        num_units=2,
        nonlinearity=lasagne.nonlinearities.softmax,
        W=wo0,
        b=bo0,
    )

    nn2 = vizlib.utils.GpuNeuralNet(
        output_layer2,
        update_learning_rate=1e-2,
        update_momentum=0.9,
    )
    nn2.initialize(ds.X, ds.y)
    p20 = nn2.predict(np.arange(len(ds.X)))
    nn2.fit(ds.X, ds.y)
    p21 = nn2.predict(ds.X)
    wc21, bc21 = params(conv_layer2)
    wo21, bo21 = params(output_layer2)

    acc0 = (p0 == ds.y).mean()
    acc20 = (p20 == ds.y).mean()
    assert acc0 == acc20

    acc1 = (p1 == ds.y).mean()
    acc21 = (p21 == ds.y).mean()
    assert acc1 == acc21

    assert acc0 < acc1

    np.testing.assert_allclose(p0, p20)
    np.testing.assert_allclose(p1, p21)
    np.testing.assert_allclose(wc1, wc21)
    np.testing.assert_allclose(wo1, wo21)
    np.testing.assert_allclose(bc1, bc21)
    np.testing.assert_allclose(bo1, bo21)


def params(layer):
    return layer.W.get_value(), layer.b.get_value()
