__author__ = 'rihards'

import layers
import theano
from theano import tensor as T
import numpy as np
import cPickle, gzip


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


def _shared_dataset(data_xy):
    """ Function that loads the dataset into shared variables

    The reason we store our dataset in shared variables is to allow
    Theano to copy it into the GPU memory (when code is run on GPU).
    Since copying data into the GPU is slow, copying a minibatch everytime
    is needed (the default behaviour if the data is not in a shared
    variable) would lead to a large decrease in performance.
    """
    data_x, data_y = data_xy
    shared_x = theano.shared(np.asarray(data_x, dtype=theano.config.floatX))
    shared_y = theano.shared(np.asarray(data_y, dtype=theano.config.floatX))
    # When storing data on the GPU it has to be stored as floats
    # therefore we will store the labels as ``floatX`` as well
    # (``shared_y`` does exactly that). But during our computations
    # we need them as ints (we use labels as index, and if they are
    # floats it doesn't make sense) therefore instead of returning
    # ``shared_y`` we will have to cast it to int. This little hack
    # lets us get around this issue
    return shared_x, T.cast(shared_y, 'int32')


def test_mnist():
    # Load the dataset
    f = gzip.open('data/mnist.pkl.gz', 'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()

    test_set_x, test_set_y = test_set
    valid_set_x, valid_set_y = valid_set
    train_set_x, train_set_y = train_set

    test_set_x = np.asmatrix(test_set_x, dtype=theano.config.floatX)
    train_set_x = np.asmatrix(train_set_x, dtype=theano.config.floatX)

    #test_set_y_vect = [[int(b) for b in list("{0:010b}".format(1 << num))[::-1]] for num in test_set_y]
    train_set_y_vect = np.asmatrix([[int(b) for b in list("{0:010b}".format(1 << num))[::-1]] for num in train_set_y], dtype=theano.config.floatX)
    #valid_set_y_vect = [[int(b) for b in list("{0:010b}".format(1 << num))[::-1]] for num in valid_set_y]

    batch_size = 500    # size of the minibatch

    # accessing the third minibatch of the training set



    import csv

    alphas = [0.001 * 3 ** i for i in range(10)]
    with open('losses.csv', 'wb') as f:
        writer = csv.writer(f)
        for alpha in alphas:
            layer1 = layers.FlatInputLayer(batch_size, test_set_x.shape[1], ranges=np.asarray([[0, 255]], dtype=theano.config.floatX))
            layer2 = layers.DenseLayer(layer1, 500, 0.1, 0, layers.sigmoid)
            layer3 = layers.DenseLayer(layer2, 10, 0.1, 0, layers.sigmoid)
            layer4 = layers.OutputLayer(layer3)

            mlp = NN([layer1, layer2, layer3, layer4], learning_rate=alpha, L2_reg=1)

            train_losses = mlp.train_model_batch(train_set_x, train_set_y_vect, epochs=50)
            print(alpha)
            print(train_losses)
            writer.writerow(train_losses)
            probabilities = mlp.fprop(test_set_x)
            predicted_labels = np.argmax(probabilities, 1)
            miss = sum([y1 == y2 for y1, y2 in zip(predicted_labels, test_set_y)])
            print(float(miss)/len(predicted_labels))


if __name__ == "__main__":
    # layer1 = layers.FlatInputLayer(32, 2)
    # layer2 = layers.DenseLayer(layer1, 60, 0.1, 0, layers.sigmoid)
    # layer3 = layers.DenseLayer(layer2, 4, 0.1, 0, layers.sigmoid)
    # layer4 = layers.OutputLayer(layer3)
    # mlp = NN([layer1, layer2, layer3, layer4])
    # x = np.matrix([0.5, 0.3], dtype='float32')
    #
    # losses = []
    #
    # # for i in xrange(3000):
    # #      X = np.matrix(np.random.rand(12000, 2), dtype='float32')
    # #      Y = np.dot(X, np.matrix([[1, 2, 0, 1],[0,0,0.5,1]], dtype='float32'))
    # #      mlp.train_model(X,Y)
    #
    # X = np.matrix(np.random.rand(120000, 2), dtype='float32')
    # Y = np.dot(X, np.matrix([[1, 2, 0, 1],[0,0,0.5,1]], dtype='float32'))
    #
    # losses = mlp.train_model_batch(X, Y)
    #
    # print losses
    #
    # print mlp.fprop(x)
    test_mnist()