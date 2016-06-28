from __future__ import print_function

from pylearn2.utils.image import tile_raster_images

import numpy as np
import sys
import os
import gzip

try:
    import cPickle as pkl
except ImportError:
    import pickle as pkl

try:
    import PIL.Image as Image
except ImportError:
    import Image

import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

# theano.config.mode = 'FAST_COMPILE'
# theano.config.exception_verbosity = 'high'

rng = np.random.RandomState(42)
trng = RandomStreams(rng.randint(2**30))
sigm = lambda x: T.nnet.sigmoid(x)


def zeros_init(name, *shape):
    value = np.zeros(shape=shape, dtype=theano.config.floatX)
    return theano.shared(name=name, value=value, borrow=True)


def glorot_init(name, *shape):
    magnitude = 4 * np.sqrt(6. / sum(shape))
    value = np.asarray(rng.uniform(low=-magnitude, high=magnitude, size=shape), dtype=theano.config.floatX)
    return theano.shared(name=name, value=value, borrow=True)


# given n_vis visible units and n_hid hidden units
# representation after filters is n_rep dimensions
# there are n_filt filters
# h = <n_hid, n_filt>

def build_rbm(n_vis, n_hid, n_rep, n_filt, n_batch):
    R = glorot_init('R', n_vis, n_filt, n_rep)
    W = glorot_init('W', n_rep, n_hid)

    vbias = zeros_init('vbias', n_vis)
    hbias = zeros_init('hbias', n_filt, n_hid)

    v = T.matrix('v', dtype=theano.config.floatX)
    lr = T.scalar('lr', dtype=theano.config.floatX)
    k = T.lscalar('k')

    def gibbs_step(v_in):
        h_inner = T.tensordot(v_in, R, axes=[1, 0])
        h_inner = T.tensordot(h_inner, W, axes=[2, 0]) + hbias
        mean_h = T.exp(h_inner) / (1 + T.sum(T.exp(h_inner), axis=1, keepdims=True))
        h = trng.multinomial(pvals=mean_h.transpose(0, 2, 1), dtype=theano.config.floatX).transpose(0, 2, 1)
        rw = T.tensordot(R, W, axes=[2, 0])
        mean_v = T.nnet.sigmoid(T.tensordot(h, rw, axes=[[1, 2], [1, 2]]) + vbias)
        v_out = trng.binomial(size=mean_v.shape, n=1, p=mean_v, dtype=theano.config.floatX)
        return mean_v, v_out

    chain, updates = theano.scan(lambda v_in: gibbs_step(v_in)[1], outputs_info=[v], n_steps=k)
    v_sample = chain[-1]

    mean_v = gibbs_step(v)[0]
    # monitor = T.xlogx.xlogy0(v, mean_v) + T.xlogx.xlogy0(1 - v, 1 - mean_v)
    monitor = -T.mean(v * T.log(mean_v) + (1 - v) * T.log(1 - mean_v))

    def free_energy(v):
        repr = T.tensordot(v, R, axes=[1, 0])
        hidden_term = T.log(1 + T.exp(T.tensordot(repr, W, axes=[2, 0]) + hbias)).sum(axis=2).sum(axis=1)
        vbias_term = T.dot(v, vbias)
        return -hidden_term - vbias_term

    cost = T.mean(free_energy(v)) - T.mean(free_energy(v_sample))

    for param in [R, W, vbias, hbias]:
        gparam = T.grad(cost=cost, wrt=param, consider_constant=[v_sample])
        updates[param] = param - gparam * lr

    return k, v, lr, W, R, cost, updates, monitor

if __name__ == '__main__':
    n_vis = 28 * 28
    n_hid = 512
    n_rep = 256
    n_filt = 128
    k = 50
    batch_size = 10
    training_epochs = 15
    save_dir = 'tirbm_mnist'

    k, v, lr, W, R, cost, updates, monitor = build_rbm(n_vis, n_hid, n_rep, n_filt, batch_size)

    if 'MNIST_PATH' not in os.environ:
        print('You must set MNIST_PATH as an environment variable (pointing at mnist.pkl.gz). You can download the ' +
              'MNIST data from http://deeplearning.net/data/mnist/mnist.pkl.gz')
        sys.exit(1)
    mnist_path = os.environ['MNIST_PATH']

    get_plt = theano.function(inputs=[], outputs=T.tensordot(R, W, axes=[2, 0]).max(axis=1).T)
    image = Image.fromarray(tile_raster_images(X=get_plt(), img_shape=(28, 28), tile_shape=(10, 10), tile_spacing=(1, 1)))
    image.save(os.path.join(save_dir, 'pre_filters.png'))

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
    index = T.lscalar('index')

    train_func = theano.function(inputs=[index, lr, k],
                                 outputs=monitor,
                                 updates=updates,
                                 givens={v: train_X[index*batch_size:(index+1)*batch_size]},
                                 name='train_rbm')

    learning_rate = 0.00001
    for epoch in range(training_epochs):
        mean_cost = list()
        for batch_index in range(n_train_batches):
            frac = (n_train_batches - batch_index) * 10 / n_train_batches
            cost = train_func(batch_index, learning_rate, 8)
            mean_cost.append(cost)
            print('\r[' + '=' * (10 - frac) + '>' + ' ' * frac + '] :: (%d / %d) Cost: %f' % (batch_index, n_train_batches, cost), end='')
        print('\r[===========] :: Cost: %f' % (np.mean(mean_cost)))

        image = Image.fromarray(tile_raster_images(X=get_plt(), img_shape=(28, 28), tile_shape=(10, 10), tile_spacing=(1, 1)))
        image.save(os.path.join(save_dir, 'filters_at_epoch_%i_batch_%i.png' % (epoch, batch_index)))
