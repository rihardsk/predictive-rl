from lasagne.layers import DenseLayer
from theano.gradient import grad_clip
from theano import tensor as T


class ClippingDenseLayer(DenseLayer):
    def __init__(self, grad_clipping=1.0, *args, **kwargs):
        super(ClippingDenseLayer, self).__init__(*args, **kwargs)
        self.grad_clipping = grad_clipping

    def get_output_for(self, input, **kwargs):
        if input.ndim > 2:
            # if the input has more than two dimensions, flatten it into a
            # batch of feature vectors.
            input = input.flatten(2)
        activation = T.dot(input, self.W)
        if self.b is not None:
            activation = activation + self.b.dimshuffle('x', 0)
        if self.grad_clipping:
            activation = grad_clip(activation, -self.grad_clipping, self.grad_clipping)
        return self.nonlinearity(activation)
