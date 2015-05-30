__author__ = 'rihards'

import layers
import theano
from theano import tensor as T
import numpy as np


class NN():
    def __init__(self, nn_layers=None, L2_reg=0.0001, learning_rate=0.01, batch_size=32):
        self.layers = nn_layers
        #TODO: maybe initialize layers and set all inputs as prev outputs

        self._batch_size = batch_size

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

        self._idx = T.lscalar('idx')

        self.x_shared = theano.shared(
            np.zeros((1, 1), dtype=theano.config.floatX))
        self.y_shared = theano.shared(
            np.zeros((1, 1), dtype=theano.config.floatX))

        self._givens = {
            self.layers[0].input_var: self.x_shared[self._idx * self._batch_size:(self._idx+1)*self._batch_size, :],
            self.layers[-1].target_var: self.y_shared[self._idx * self._batch_size:(self._idx+1)*self._batch_size, :],
        }

        self._train_model_batch = theano.function(
            inputs=[self._idx],
            updates=self.updates,
            givens=self._givens,
            outputs=self.cost
        )

    def output(self, *args, **kwargs):
        last_layer_output = self.layers[-1].output(*args, **kwargs)
        return last_layer_output

    def fprop(self, x):
        return self._fprop(x)

    def train_model_batch(self, X, Y, epochs=20):
        num_batches_valid = X.shape[0] // self._batch_size
        self.x_shared.set_value(X)
        self.y_shared.set_value(Y)
        epoch_losses = []
        for epoch in xrange(epochs):
            losses = []
            for b in xrange(num_batches_valid):
                loss = self._train_model_batch(b)
                losses.append(loss)
            mean_train_loss = np.sqrt(np.mean(losses))
            epoch_losses.append(mean_train_loss)
        return epoch_losses

if __name__ == "__main__":
    layer1 = layers.FlatInputLayer(32, 2)
    layer2 = layers.DenseLayer(layer1, 60, 0.1, 0, layers.sigmoid)
    layer3 = layers.DenseLayer(layer2, 4, 0.1, 0, layers.sigmoid)
    layer4 = layers.OutputLayer(layer3)
    mlp = NN([layer1, layer2, layer3, layer4])
    x = np.matrix([0.5, 0.3], dtype='float32')

    losses = []

    # for i in xrange(3000):
    #      X = np.matrix(np.random.rand(12000, 2), dtype='float32')
    #      Y = np.dot(X, np.matrix([[1, 2, 0, 1],[0,0,0.5,1]], dtype='float32'))
    #      mlp.train_model(X,Y)

    X = np.matrix(np.random.rand(120000, 2), dtype='float32')
    Y = np.dot(X, np.matrix([[1, 2, 0, 1],[0,0,0.5,1]], dtype='float32'))

    losses = mlp.train_model_batch(X, Y)

    print losses

    print mlp.fprop(x)