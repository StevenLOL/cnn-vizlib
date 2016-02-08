'''
'''
import lasagne
import theano

class Deconv2DLayer(lasagne.layers.Layer):

    def __init__(self, incoming, **kwargs):
        super(Deconv2DLayer, self).__init__(incoming, **kwargs)

        self.nonlinearity = incoming.nonlinearity
        if self.nonlinearity != lasagne.nonlinearities.rectify:
            raise NotImplementedError()

        self.num_filters = incoming.num_filters
        self.filter_size = incoming.filter_size[::-1]
        self.stride = incoming.stride
        self.convolution = incoming.convolution

        self.pad = incoming.pad
        if self.pad != 'same':
            raise NotImplementedError()

        # Unfortunately W can not be shared.
        # This means that IF the convolution network is trained after the
        # deconvolution part is attached, they will get out of sync.
        # I don't think I have a means of checking this, since get_output_for
        # should compile to an expression.
        self.W = theano.shared(incoming.W.get_value().T, borrow=False)

        # TODO: figure out how to handle the bias
        # options:
        # 1) Set the bias to 0 [ignore it]
        # 2) Subtract before convolution
        # 3) Subtract after convolution
        # for now we only implement option 0.

    def get_W_shape(self):
        return lasagne.layers.Conv2DLayer.get_W_shape(self)

    def get_output_shape_for(self, input_shape):
        return lasagne.layers.Conv2DLayer.get_output_shape_for(self)

    def get_output_for(self, input, input_shape=None, **kwargs):
        if input_shape is None:
            input_shape = self.input_shape

        # should apply the nonlinearity BEFORE deconvoluting
        rectified = self.nonlinearity(input)

        # simulate same convolution by cropping a full convolution
        conved = self.convolution(
            rectified, self.W, subsample=self.stride,
            image_shape=input_shape,
            filter_shape=self.get_W_shape(),
            border_mode='full'
        )
        shift_x = (self.filter_size[0] - 1) // 2
        shift_y = (self.filter_size[1] - 1) // 2
        conved = conved[:, :, shift_x:rectified.shape[2] + shift_x,
                        shift_y:rectified.shape[3] + shift_y]

        return conved
