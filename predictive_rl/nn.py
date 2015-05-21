__author__ = 'rihards'

import layers
import theano
from theano import tensor as T
import numpy as np


class NN():
    def __init__(self, nn_layers=None, L2_reg=0.0001, learning_rate=0.01):
        self.layers = nn_layers
        #TODO: maybe initialize layers and set all inputs as prev outputs

        self._fprop = theano.function(
            [self.layers[0].input_var],
            self.output()
        )

        self.parameters = layers.all_parameters(self.layers[-1])

        self.cost = self.layers[-1].error()

        self.regularization = sum([(W_or_b ** 2).sum() for W_or_b in self.parameters])

        self.updates = layers.gen_updates_sgd(self.cost + self.regularization * L2_reg, self.parameters, learning_rate) # the last layer must be a layers.OutputLayer

        self.train_model = theano.function(
            inputs=[self.layers[0].input_var, self.layers[-1].target_var],
            updates=self.updates,
            outputs=self.cost
        )

    def output(self, *args, **kwargs):
        last_layer_output = self.layers[-1].output(*args, **kwargs)
        return last_layer_output

    def fprop(self, x):
        return self._fprop(x)

if __name__ == "__main__":
    layer1 = layers.FlatInputLayer(2, 2)
    layer2 = layers.DenseLayer(layer1, 60, 0.1, 0, layers.sigmoid)
    layer3 = layers.DenseLayer(layer2, 4, 0.1, 0, layers.sigmoid)
    layer4 = layers.OutputLayer(layer3)
    mlp = NN([layer1, layer2, layer3, layer4])
    x = np.matrix([0.5, 0.3], dtype='float32')

    for i in xrange(3000):
         X = np.matrix(np.random.rand(12000, 2), dtype='float32')
         Y = np.dot(X, np.matrix([[1, 2, 0, 1],[0,0,0.5,1]], dtype='float32'))
         mlp.train_model(X,Y)

    print mlp.fprop(x)