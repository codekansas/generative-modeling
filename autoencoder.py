from __future__ import print_function

import gzip

from pylearn2.utils.image import tile_raster_images
from theano.tensor.shared_randomstreams import RandomStreams

try:
    import PIL.Image as Image
except ImportError:
    import Image

import theano
import theano.tensor as T
import numpy as np
import os
import sys
import timeit

import cPickle as pkl

rng = np.random.RandomState(42)
trng = RandomStreams(rng.randint(2**30))


def activation(x):
    return T.nnet.sigmoid(x)


def get_weight(name, *shape, **kwargs):
    magnitude = kwargs.get('magnitude', 1)
    value = np.asarray(rng.uniform(low=-magnitude, high=magnitude, size=shape), dtype=theano.config.floatX)
    return theano.shared(name=name, value=value, borrow=True)


def get_zeros(name, *shape):
    value = np.zeros(*shape, dtype=theano.config.floatX)
    return theano.shared(name=name, value=value, borrow=True)


def get_cost(input, output):
    return -(input * T.log(output) + (1. - input) * T.log(1. - output)).sum(axis=1).mean(axis=0)


# def get_cost(input, output):
#     return ((input - output) ** 2).sum(axis=0).mean() // 2.


def make_autoencoder(input, lr, n_dim, n_hidden):
    magnitude = 4 * np.sqrt(6. / (n_hidden + n_dim))
    W = get_weight('W', n_dim, n_hidden, magnitude=magnitude)

    b_hid = get_zeros('b_hid', n_hidden)
    b_vis = get_zeros('b_vis', n_dim)

    input_corrupted = trng.binomial(size=input.shape, p=0.7, dtype=theano.config.floatX) * input

    hidden = activation(T.dot(input_corrupted, W) + b_hid)
    output = activation(T.dot(hidden, W.T) + b_vis)

    cost = get_cost(input, output)
    updates = [(p, p - lr * T.grad(cost, p)) for p in [W, b_hid, b_vis]]

    return W, cost, updates, output

if __name__ == '__main__':

    training_epochs = 15
    batch_size = 50
    n_hidden = 500
    tile_shape = (10, 10)
    save_dir = 'mnist_autoencoder'

    if 'MNIST_PATH' not in os.environ:
        print('You must set MNIST_PATH as an environment variable (pointing at mnist.pkl.gz). You can download the ' +
              'MNIST data from http://deeplearning.net/data/mnist/mnist.pkl.gz')
        sys.exit(1)
    mnist_path = os.environ['MNIST_PATH']

    f = gzip.open(mnist_path, 'rb')
    train_set, _, test_set = pkl.load(f)
    f.close()

    def split_shared(xy):
        x, y = xy
        x = theano.shared(np.asarray(x, dtype=theano.config.floatX))
        y = theano.shared(np.asarray(y, dtype=theano.config.floatX))
        return x, T.cast(y, 'int32')

    train_X, train_y = split_shared(train_set)
    test_X, test_y = split_shared(test_set)

    n_train_batches = train_X.get_value(borrow=True).shape[0] // batch_size
    index = T.lscalar()
    lr = T.dscalar('lr')
    x = T.dmatrix('x')

    W, cost, updates, output = make_autoencoder(x, lr, 28*28, n_hidden)

    train_autoencoder = theano.function(inputs=[index, lr],
                                        outputs=cost,
                                        updates=updates,
                                        givens={x: train_X[index * batch_size: (index + 1) * batch_size]},
                                        name='train_autoencoder')

    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    os.chdir(save_dir)

    plotting_time = 0.
    pretrain_time = 0.

    lrv = 0.1

    for epoch in range(training_epochs):
        start_time = timeit.default_timer()
        mean_cost = list()
        for batch_index in range(n_train_batches):
            cost = train_autoencoder(batch_index, lrv)
            mean_cost.append(cost)
            frac = (n_train_batches - batch_index) * 10 / n_train_batches
            print('\r[' + '=' * (10 - frac) + '>' + ' ' * frac + '] Cost: %f' % cost, end='')
        stop_time = timeit.default_timer()
        pretrain_time += (stop_time - start_time)

        print('\r[===========] Training epoch %d, took %f seconds, cost is' % (epoch, stop_time - start_time), np.mean(mean_cost))

        plotting_start = timeit.default_timer()
        image = Image.fromarray(tile_raster_images(X=W.get_value(borrow=True).T, img_shape=(28, 28), tile_shape=tile_shape, tile_spacing=(1, 1)))

        image.save('filters_at_epoch_%i.png' % epoch)
        plotting_stop = timeit.default_timer()
        plotting_time += (plotting_stop - plotting_start)

    print('Training took %f minutes' % (pretrain_time / 60.))
