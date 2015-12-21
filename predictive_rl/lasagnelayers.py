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
        clipped_W = self.W
        clipped_b = self.b
        if self.grad_clipping:
            clipped_W = grad_clip(clipped_W, -self.grad_clipping, self.grad_clipping)
            if clipped_b is not None:
                clipped_b = grad_clip(clipped_b, -self.grad_clipping, self.grad_clipping)
        activation = T.dot(input, clipped_W)
        if clipped_b is not None:
            activation = activation + clipped_b.dimshuffle('x', 0)
        return self.nonlinearity(activation)
