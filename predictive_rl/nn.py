import os
from matplotlib._delaunay import nn_interpolate_grid
import sys
from theano.tensor.signal import downsample

__author__ = 'rihards'

import layers
import theano
from theano import tensor as T
import numpy as np
import cPickle, gzip
import time
from theano.tensor.nnet import conv


def load_data(dataset):
    ''' Loads the dataset

    :type dataset: string
    :param dataset: the path to the dataset (here MNIST)
    '''

    #############
    # LOAD DATA #
    #############

    # Download the MNIST dataset if it is not present
    data_dir, data_file = os.path.split(dataset)
    if data_dir == "" and not os.path.isfile(dataset):
        # Check if dataset is in the data directory.
        new_path = os.path.join(
            os.path.split(__file__)[0],
            "data",
            dataset
        )
        if os.path.isfile(new_path) or data_file == 'mnist.pkl.gz':
            dataset = new_path

    if (not os.path.isfile(dataset)) and data_file == 'mnist.pkl.gz':
        import urllib
        origin = (
            'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
        )
        print 'Downloading data from %s' % origin
        urllib.urlretrieve(origin, dataset)

    print '... loading data'

    # Load the dataset
    f = gzip.open(dataset, 'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()
    #train_set, valid_set, test_set format: tuple(input, target)
    #input is an numpy.ndarray of 2 dimensions (a matrix)
    #witch row's correspond to an example. target is a
    #numpy.ndarray of 1 dimensions (vector)) that have the same length as
    #the number of rows in the input. It should give the target
    #target to the example with the same index in the input.

    def shared_dataset(data_xy, borrow=True):
        """ Function that loads the dataset into shared variables

        The reason we store our dataset in shared variables is to allow
        Theano to copy it into the GPU memory (when code is run on GPU).
        Since copying data into the GPU is slow, copying a minibatch everytime
        is needed (the default behaviour if the data is not in a shared
        variable) would lead to a large decrease in performance.
        """
        data_x, data_y = data_xy
        shared_x = theano.shared(np.asarray(data_x,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        shared_y = theano.shared(np.asarray(data_y,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        # When storing data on the GPU it has to be stored as floats
        # therefore we will store the labels as ``floatX`` as well
        # (``shared_y`` does exactly that). But during our computations
        # we need them as ints (we use labels as index, and if they are
        # floats it doesn't make sense) therefore instead of returning
        # ``shared_y`` we will have to cast it to int. This little hack
        # lets ous get around this issue
        return shared_x, T.cast(shared_y, 'int32')

    test_set_x, test_set_y = shared_dataset(test_set)
    valid_set_x, valid_set_y = shared_dataset(valid_set)
    train_set_x, train_set_y = shared_dataset(train_set)

    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
            (test_set_x, test_set_y)]
    return rval


class LeNetConvPoolLayer(object):
    """Pool Layer of a convolutional network """

    def __init__(self, rng, input, filter_shape, image_shape, poolsize=(2, 2)):
        """
        Allocate a LeNetConvPoolLayer with shared variable internal parameters.

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dtensor4
        :param input: symbolic image tensor, of shape image_shape

        :type filter_shape: tuple or list of length 4
        :param filter_shape: (number of filters, num input feature maps,
                              filter height, filter width)

        :type image_shape: tuple or list of length 4
        :param image_shape: (batch size, num input feature maps,
                             image height, image width)

        :type poolsize: tuple or list of length 2
        :param poolsize: the downsampling (pooling) factor (#rows, #cols)
        """

        assert image_shape[1] == filter_shape[1]
        self.input = input

        # there are "num input feature maps * filter height * filter width"
        # inputs to each hidden unit
        fan_in = np.prod(filter_shape[1:])
        # each unit in the lower layer receives a gradient from:
        # "num output feature maps * filter height * filter width" /
        #   pooling size
        fan_out = (filter_shape[0] * np.prod(filter_shape[2:]) /
                   np.prod(poolsize))
        # initialize weights with random weights
        W_bound = np.sqrt(6. / (fan_in + fan_out))
        self.W = theano.shared(
            np.asarray(
                rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
                dtype=theano.config.floatX
            ),
            borrow=True
        )

        # the bias is a 1D tensor -- one bias per output feature map
        b_values = np.zeros((filter_shape[0],), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, borrow=True)

        # convolve input feature maps with filters
        conv_out = conv.conv2d(
            input=input,
            filters=self.W,
            filter_shape=filter_shape,
            image_shape=image_shape
        )

        # downsample each feature map individually, using maxpooling
        pooled_out = downsample.max_pool_2d(
            input=conv_out,
            ds=poolsize,
            ignore_border=True
        )

        # add the bias term. Since the bias is a vector (1D array), we first
        # reshape it to a tensor of shape (1, n_filters, 1, 1). Each bias will
        # thus be broadcasted across mini-batches and feature map
        # width & height
        self.output = T.tanh(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))

        # store parameters of this layer
        self.params = [self.W, self.b]


class HiddenLayer(object):
    def __init__(self, rng, input, n_in, n_out, W=None, b=None,
                 activation=T.tanh):
        """
        Typical hidden layer of a MLP: units are fully-connected and have
        sigmoidal activation function. Weight matrix W is of shape (n_in,n_out)
        and the bias vector b is of shape (n_out,).

        NOTE : The nonlinearity used here is tanh

        Hidden unit activation is given by: tanh(dot(input,W) + b)

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dmatrix
        :param input: a symbolic tensor of shape (n_examples, n_in)

        :type n_in: int
        :param n_in: dimensionality of input

        :type n_out: int
        :param n_out: number of hidden units

        :type activation: theano.Op or function
        :param activation: Non linearity to be applied in the hidden
                           layer
        """
        self.input = input
        # end-snippet-1

        # `W` is initialized with `W_values` which is uniformely sampled
        # from sqrt(-6./(n_in+n_hidden)) and sqrt(6./(n_in+n_hidden))
        # for tanh activation function
        # the output of uniform if converted using asarray to dtype
        # theano.config.floatX so that the code is runable on GPU
        # Note : optimal initialization of weights is dependent on the
        #        activation function used (among other things).
        #        For example, results presented in [Xavier10] suggest that you
        #        should use 4 times larger initial weights for sigmoid
        #        compared to tanh
        #        We have no info for other function, so we use the same as
        #        tanh.
        if W is None:
            W_values = np.asarray(
                rng.uniform(
                    low=-np.sqrt(6. / (n_in + n_out)),
                    high=np.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)
                ),
                dtype=theano.config.floatX
            )
            if activation == theano.tensor.nnet.sigmoid:
                W_values *= 4

            W = theano.shared(value=W_values, name='W', borrow=True)

        if b is None:
            b_values = np.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)

        self.W = W
        self.b = b

        lin_output = T.dot(input, self.W) + self.b
        self.output = (
            lin_output if activation is None
            else activation(lin_output)
        )
        # parameters of the model
        self.params = [self.W, self.b]


class LogisticRegression(object):
    """Multi-class Logistic Regression Class

    The logistic regression is fully described by a weight matrix :math:`W`
    and bias vector :math:`b`. Classification is done by projecting data
    points onto a set of hyperplanes, the distance to which is used to
    determine a class membership probability.
    """

    def __init__(self, input, n_in, n_out):
        """ Initialize the parameters of the logistic regression

        :type input: theano.tensor.TensorType
        :param input: symbolic variable that describes the input of the
                      architecture (one minibatch)

        :type n_in: int
        :param n_in: number of input units, the dimension of the space in
                     which the datapoints lie

        :type n_out: int
        :param n_out: number of output units, the dimension of the space in
                      which the labels lie

        """
        # start-snippet-1
        # initialize with 0 the weights W as a matrix of shape (n_in, n_out)
        self.W = theano.shared(
            value=np.zeros(
                (n_in, n_out),
                dtype=theano.config.floatX
            ),
            name='W',
            borrow=True
        )
        # initialize the baises b as a vector of n_out 0s
        self.b = theano.shared(
            value=np.zeros(
                (n_out,),
                dtype=theano.config.floatX
            ),
            name='b',
            borrow=True
        )

        # symbolic expression for computing the matrix of class-membership
        # probabilities
        # Where:
        # W is a matrix where column-k represent the separation hyper plain for
        # class-k
        # x is a matrix where row-j  represents input training sample-j
        # b is a vector where element-k represent the free parameter of hyper
        # plain-k
        self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)

        # symbolic description of how to compute prediction as class whose
        # probability is maximal
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)
        # end-snippet-1

        # parameters of the model
        self.params = [self.W, self.b]

    def negative_log_likelihood(self, y):
        """Return the mean of the negative log-likelihood of the prediction
        of this model under a given target distribution.

        .. math::

            \frac{1}{|\mathcal{D}|} \mathcal{L} (\theta=\{W,b\}, \mathcal{D}) =
            \frac{1}{|\mathcal{D}|} \sum_{i=0}^{|\mathcal{D}|}
                \log(P(Y=y^{(i)}|x^{(i)}, W,b)) \\
            \ell (\theta=\{W,b\}, \mathcal{D})

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label

        Note: we use the mean instead of the sum so that
              the learning rate is less dependent on the batch size
        """
        # start-snippet-2
        # y.shape[0] is (symbolically) the number of rows in y, i.e.,
        # number of examples (call it n) in the minibatch
        # T.arange(y.shape[0]) is a symbolic vector which will contain
        # [0,1,2,... n-1] T.log(self.p_y_given_x) is a matrix of
        # Log-Probabilities (call it LP) with one row per example and
        # one column per class LP[T.arange(y.shape[0]),y] is a vector
        # v containing [LP[0,y[0]], LP[1,y[1]], LP[2,y[2]], ...,
        # LP[n-1,y[n-1]]] and T.mean(LP[T.arange(y.shape[0]),y]) is
        # the mean (across minibatch examples) of the elements in v,
        # i.e., the mean log-likelihood across the minibatch.
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])
        # end-snippet-2

    def errors(self, y):
        """Return a float representing the number of errors in the minibatch
        over the total number of examples of the minibatch ; zero one
        loss over the size of the minibatch

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label
        """

        # check if y has same dimension of y_pred
        if y.ndim != self.y_pred.ndim:
            raise TypeError(
                'y should have the same shape as self.y_pred',
                ('y', y.type, 'y_pred', self.y_pred.type)
            )
        # check if y is of the correct datatype
        if y.dtype.startswith('int'):
            # the T.neq operator returns a vector of 0s and 1s, where 1
            # represents a mistake in prediction
            return T.mean(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError()



class NN():
    def __init__(self, train_set_x, train_set_y, nn_layers=None, L2_reg=0.0001, learning_rate=0.1, batch_size=32):
        nn_layers = []
        nkerns = [20, 50]
        # nn_layers.append(layers.Input2DLayer(batch_size, 1, 28, 28, scale=255))

        x = T.matrix('x')   # the data is presented as rasterized images
        y = T.ivector('y')  # the labels are presented as 1D vector of
        """
        nn_layers[0].input_var = x.reshape((batch_size, 1, 28, 28))
        nn_layers.append(layers.Conv2DLayer(nn_layers[-1], nkerns[0], 5, 5, .01, .01))
        nn_layers.append(layers.Pooling2DLayer(nn_layers[-1], pool_size=(2, 2)))
        nn_layers.append(layers.Conv2DLayer(nn_layers[-1], nkerns[1], 5, 5, .01, .01))
        nn_layers.append(layers.Pooling2DLayer(nn_layers[-1], pool_size=(2, 2)))
        #nn_layers.append(layers.FlattenLayer(nn_layers[-1]))
        nn_layers.append(layers.DenseLayer(nn_layers[-1], 500, 0.1, 0, nonlinearity=layers.tanh))
        nn_layers.append(layers.SoftmaxLayer(nn_layers[-1], 10, 0.1, 0, nonlinearity=layers.tanh))
        #nn_layers.append(layers.OutputLayer(nn_layers[-1]))
        """

        self.layers = nn_layers
        #TODO: maybe initialize layers and set all inputs as prev outputs

        self._batch_size = batch_size

        rng = np.random.RandomState(23455)
        # layer0_input = x.reshape((batch_size, 1, 28, 28))

        nn_layers.append(layers.Input2DLayer(batch_size, 1, 28, 28))
        nn_layers[0].input_var = x.reshape((batch_size, 1, 28, 28))
        nn_layers.append(layers.StridedConv2DLayer(nn_layers[-1],
                                                     n_filters=nkerns[0],
                                                     filter_width=5,
                                                     filter_height=5,
                                                     stride_x=2,
                                                     stride_y=2,
                                                     weights_std=.01,
                                                     init_bias_value=0.01,
                                                     nonlinearity=T.tanh))
        # nn_layers.append(layers.Conv2DLayer(nn_layers[-1], nkerns[0], 5, 5, .01, .01, nonlinearity=T.tanh))
        # nn_layers.append(layers.Pooling2DLayer(nn_layers[-1], pool_size=(2, 2)))


        # nn_layers.append(LeNetConvPoolLayer(
        #     rng,
        #     input=nn_layers[-1].output(),
        #     image_shape=(batch_size, 1, 28, 28),
        #     filter_shape=(nkerns[0], 1, 5, 5),
        #     poolsize=(2, 2)
        # ))

        nn_layers.append(layers.StridedConv2DLayer(nn_layers[-1],
                                                     n_filters=nkerns[1],
                                                     filter_width=5,
                                                     filter_height=5,
                                                     stride_x=2,
                                                     stride_y=2,
                                                     weights_std=.01,
                                                     init_bias_value=0.01,
                                                     nonlinearity=T.tanh))
        # nn_layers.append(layers.Conv2DLayer(nn_layers[-1], nkerns[1], 5, 5, .01, .01))
        # nn_layers.append(layers.Pooling2DLayer(nn_layers[-1], pool_size=(2, 2)))

        # nn_layers.append(LeNetConvPoolLayer(
        #     rng,
        #     input=nn_layers[-1].output(),
        #     image_shape=(batch_size, nkerns[0], 12, 12),
        #     filter_shape=(nkerns[1], nkerns[0], 5, 5),
        #     poolsize=(2, 2)
        # ))


        nn_layers.append(layers.DenseLayer(nn_layers[-1], 500, 0.1, 0, nonlinearity=layers.tanh))

        # layer2_input = nn_layers[-1].output().flatten(2)
        # nn_layers.append(HiddenLayer(
        #     rng,
        #     input=layer2_input,
        #     n_in=nkerns[1] * 4 * 4,
        #     n_out=500,
        #     activation=T.tanh
        # ))

        nn_layers.append(layers.SoftmaxLayer(nn_layers[-1], 10, 0.1, 0, nonlinearity=layers.tanh))

        # nn_layers.append(LogisticRegression(input=nn_layers[-1].output(), n_in=500, n_out=10))


        # self._fprop = theano.function(
        #     [self.layers[0].input_var],
        #     self.output()
        # )

        self.parameters = layers.all_parameters(self.layers[-1])
        # self.parameters = [param for layer in nn_layers[1:] for param in layer.params] #nn_layers[5].params + nn_layers[4].params + nn_layers[3].params + nn_layers[2].params + nn_layers[1].params

        #self.cost = self.layers[-1].error()
        self.cost = self.layers[-1].negative_log_likelihood(y)

        # grads = T.grad(cost, self.parameters)

        self.regularization = sum([(W_or_b ** 2).sum() for W_or_b in self.parameters])

        self.updates = layers.gen_updates_sgd(self.cost + self.regularization * L2_reg, self.parameters, learning_rate) # the last layer must be a layers.OutputLayer
        # self.updates = layers.gen_updates_sgd(cost, self.parameters, learning_rate) # the last layer must be a layers.OutputLayer

        # self.updates = [
        #     (param_i, param_i - learning_rate * grad_i)
        #     for param_i, grad_i in zip(self.parameters, grads)
        # ]

        # theano.pp(self.updates)
        # self.train_model = theano.function(
        #     inputs=[self.layers[0].input_var, self.layers[-1].target_var],
        #     updates=self.updates,
        #     outputs=self.cost
        # )

        self._idx = T.lscalar('idx')
        # self.x_shared = theano.shared(
        #     np.zeros(self.layers[0].get_output_shape(), dtype=theano.config.floatX))
        # self.y_shared = theano.shared(
        #     np.zeros(self.layers[-1].get_output_shape(), dtype=self.layers[-1].output().dtype))

        self._givens = {
            x: train_set_x[self._idx * self._batch_size: (self._idx+1)*self._batch_size],
            y: train_set_y[self._idx * self._batch_size: (self._idx+1)*self._batch_size],
        }

        self._train_model_batch = theano.function(
            inputs=[self._idx],
            updates=self.updates,
            givens=self._givens,
            outputs=self.cost
        )

        # self._output_model_batch = theano.function(
        #     inputs=[self._idx],
        #     updates=self.updates,
        #     givens=self._givens,
        #     outputs=self.output()
        # )

        #self._test_model = theano.function(
        #    [self.layers[0].input_var, self.layers[-1]],
        #    self.cost
        #)

    # def output(self, *args, **kwargs):
    #     last_layer_output = self.layers[-1].output(*args, **kwargs)
    #     return last_layer_output

    # def fprop(self, x):
    #     return self._fprop(x)

    def train_model_batch(self, X, Y, epochs=20):
        num_batches_valid = X.get_value(borrow=True).shape[0] // self._batch_size
        # self.x_shared.set_value(X)
        # self.y_shared.set_value(Y)
        epoch_losses = []
        for epoch in xrange(epochs):
            losses = []
            for b in xrange(num_batches_valid):
                loss = self._train_model_batch(b)
                losses.append(loss)
                print >> sys.stderr, ('\tEpoch %i\tBatch %i/%i\tLoss %f' % (epoch, b, num_batches_valid, loss))
            #mean_train_loss = np.sqrt(np.mean(losses))
            mean_train_loss = np.mean(losses)
            print >> sys.stderr, ('Epoch %i\tAverage loss %f' % (epoch, mean_train_loss))
            epoch_losses.append(mean_train_loss)
        return epoch_losses


    # def output_model_batch(self, X):
    #     num_batches_valid = X.shape[0] // self._batch_size
    #     self.x_shared.set_value(X)
    #     outputs = []
    #     for b in xrange(num_batches_valid):
    #         batch_outputs = self._output_model_batch(b)
    #         outputs.append(batch_outputs)
    #     return np.concatenate(outputs)


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


def test_convnet():
    batch_size = 500    # size of the minibatch
    learning_rate = 0.1
    n_epochs = 200

    # Load the dataset
    #f = gzip.open('data/mnist.pkl.gz', 'rb')
    train_set, valid_set, test_set = load_data('mnist.pkl.gz') #cPickle.load(f)
    # f.close()

    test_set_x, test_set_y = test_set
    valid_set_x, valid_set_y = valid_set
    train_set_x, train_set_y = train_set

    # test_set_x = np.asarray(test_set_x, dtype=theano.config.floatX)
    # train_set_x = np.asarray(train_set_x, dtype=theano.config.floatX)
    # train_set_y = np.asarray(train_set_y, dtype='int32')

    # test_set_y_vect = [[int(b) for b in list("{0:010b}".format(1 << num))[::-1]] for num in test_set_y]
    # train_set_y_vect = np.asmatrix([[int(b) for b in list("{0:010b}".format(1 << num))[::-1]] for num in train_set_y], dtype=theano.config.floatX)
    # valid_set_y_vect = [[int(b) for b in list("{0:010b}".format(1 << num))[::-1]] for num in valid_set_y]


    # train_set_x = train_set_x.reshape((train_set_x.shape[0], 1, 28, 28))
    # test_set_x = test_set_x.reshape((test_set_x.shape[0], 1, 28, 28))
    # train_set_y = train_set_y.reshape((train_set_y.shape[0], 1))

    # compute number of minibatches for training, validation and testing
    """
    n_train_batches = train_set_x.get_value(borrow=True).shape[0]
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]
    n_test_batches = test_set_x.get_value(borrow=True).shape[0]
    n_train_batches /= batch_size
    n_valid_batches /= batch_size
    n_test_batches /= batch_size
    """


    # nn_layers = []
    # nkerns = [20, 50]
    # nn_layers.append(layers.Input2DLayer(batch_size, 1, 28, 28, scale=255))
    # nn_layers.append(layers.Conv2DLayer(nn_layers[-1], nkerns[0], 5, 5, .01, .01))
    # nn_layers.append(layers.Pooling2DLayer(nn_layers[-1], pool_size=(2, 2)))
    # nn_layers.append(layers.Conv2DLayer(nn_layers[-1], nkerns[1], 5, 5, .01, .01))
    # nn_layers.append(layers.Pooling2DLayer(nn_layers[-1], pool_size=(2, 2)))
    # #nn_layers.append(layers.FlattenLayer(nn_layers[-1]))
    # nn_layers.append(layers.DenseLayer(nn_layers[-1], 500, 0.1, 0, nonlinearity=layers.tanh))
    # nn_layers.append(layers.SoftmaxLayer(nn_layers[-1], 10, 0.1, 0, nonlinearity=layers.tanh))
    # #nn_layers.append(layers.OutputLayer(nn_layers[-1]))

    mlp = NN(train_set_x, train_set_y, batch_size=batch_size)
    """
    ###############
    # TRAIN MODEL #
    ###############
    print '... training'
    # early-stopping parameters
    patience = 10000  # look as this many examples regardless
    patience_increase = 2  # wait this much longer when a new best is
                        # found
    improvement_threshold = 0.995  # a relative improvement of this much is
                                # considered significant
    validation_frequency = min(n_train_batches, patience / 2)
                               # go through this many
                               # minibatche before checking the network
                               # on the validation set; in this case we
                               # check every epoch

    best_validation_loss = np.inf
    best_iter = 0
    test_score = 0.
    start_time = time.clock()

    epoch = 0
    done_looping = False

    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in xrange(n_train_batches):
            iter = (epoch - 1) * n_train_batches + minibatch_index

            if iter % 100 == 0:
                print 'training @ iter = ', iter
            cost_ij = train_model(minibatch_index)


            if (iter + 1) % validation_frequency == 0:

                # compute zero-one loss on validation set
                validation_losses = [validate_model(i) for i
                                     in xrange(n_valid_batches)]
                this_validation_loss = np.mean(validation_losses)
                print('epoch %i, minibatch %i/%i, validation error %f %%' %
                      (epoch, minibatch_index + 1, n_train_batches,
                       this_validation_loss * 100.))

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:

                    #improve patience if loss improvement is good enough
                    if this_validation_loss < best_validation_loss *  \
                       improvement_threshold:
                        patience = max(patience, iter * patience_increase)

                    # save best validation score and iteration number
                    best_validation_loss = this_validation_loss
                    best_iter = iter

                    # test it on the test set
                    test_losses = [
                        test_model(i)
                        for i in xrange(n_test_batches)
                    ]
                    test_score = np.mean(test_losses)
                    print(('     epoch %i, minibatch %i/%i, test error of '
                           'best model %f %%') %
                          (epoch, minibatch_index + 1, n_train_batches,
                           test_score * 100.))

            if patience <= iter:
                done_looping = True
                break

    end_time = time.clock()
    print('Optimization complete.')
    print('Best validation score of %f %% obtained at iteration %i, '
          'with test performance %f %%' %
          (best_validation_loss * 100., best_iter + 1, test_score * 100.))
    print >> sys.stderr, ('The code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))
    """
    start_time = time.clock()
    train_losses = mlp.train_model_batch(train_set_x, train_set_y, n_epochs)
    end_time = time.clock()
    print >> sys.stderr, ('The code ran for %.2fm' % ((end_time - start_time) / 60.))
    print 'train losses'
    print train_losses
    print 'mean train loss'
    np.mean(train_losses)
    # print 'testing'
    # #test_mb_size = test_set_x.shape[0]
    # #nn_layers[0].mb_size = test_mb_size
    # #mlp_test = NN(nn_layers, batch_size=test_mb_size)
    # predicted_classes = mlp.output_model_batch(test_set_x)
    # miss = predicted_classes != test_set_y
    # test_error_rate = float(len(miss[miss])) / len(miss)
    # print test_error_rate
    print 'done'

if __name__ == "__main__":
    # layer1 = layers.FlatInputLayer(32, 2)
    # layer2 = layers.DenseLayer(layer1, 60, 0.1, 0, layers.sigmoid)
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
    test_convnet()


