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
# def sgd_with_grad_clipping(loss_or_grads, params, learning_rate, rescale):
#     grads = lasagne.updates.get_or_compute_grads(loss_or_grads, params)
#     updates = OrderedDict()
#
#     grad_norm = T.sqrt(sum(map(lambda x: T.sqr(x).sum(), grads)))
#     not_finite = T.or_(T.isnan(grad_norm), T.isinf(grad_norm))
#     grad_norm = T.sqrt(grad_norm)
#     scaling_num = rescale
#     scaling_den = T.maximum(rescale, grad_norm)
#     for n, (param, grad) in enumerate(zip(params, grads)):
#         grad = T.switch(not_finite, 0.1 * param,
#             grad * (scaling_num / scaling_den))
#         updates[param] = param - learning_rate * grad
#     return updates

# max_norm = 5.0
# grads = theano.gradient(loss, params)
# grads = [lasagne.updates.norm_constraint(grad, max_norm, range(grad.ndim))
#          for grad in grads]
# updates = lasagne.updates.whatever(grads, params)