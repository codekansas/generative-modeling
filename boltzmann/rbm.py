from __future__ import print_function

from pylearn2.utils.image import tile_raster_images
from theano.tensor.shared_randomstreams import RandomStreams

try:
    import PIL.Image as Image
except ImportError:
    import Image

import numpy as np

import theano
import theano.tensor as T
import os
import sys
import gzip
import cPickle as pkl
import timeit

# initialize random number generators
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


class RBM:
    def __init__(self, n_visible, n_hidden, input):
        self.n_visible = n_visible
        self.n_hidden = n_hidden

        magnitude = 4 * np.sqrt(6. / (n_hidden + n_visible))
        self.W = get_weight('W', n_visible, n_hidden, magnitude=magnitude)
        self.hbias = get_zeros('hbias', n_hidden)
        self.vbias = get_zeros('vbias', n_visible)
        self.input = input
        self.params = [self.W, self.hbias, self.vbias]

    def free_energy(self, v_sample):
        wx_b = T.dot(v_sample, self.W) + self.hbias
        vbias_term = T.dot(v_sample, self.vbias)
        hidden_term = T.sum(T.log(1 + T.exp(wx_b)), axis=1)
        return -hidden_term - vbias_term

    def propup(self, vis):
        pre_sigmoid_activation = T.dot(vis, self.W) + self.hbias
        return [pre_sigmoid_activation, activation(pre_sigmoid_activation)]

    def sample_h_given_v(self, v0_sample):
        pre_sigmoid_h1, h1_mean = self.propup(v0_sample)
        h1_sample = trng.binomial(size=h1_mean.shape, n=1, p=h1_mean, dtype=theano.config.floatX)
        return [pre_sigmoid_h1, h1_mean, h1_sample]

    def propdown(self, hid):
        pre_sigmoid_activation = T.dot(hid, self.W.T) + self.vbias
        return [pre_sigmoid_activation, activation(pre_sigmoid_activation)]

    def sample_v_given_h(self, h0_sample):
        pre_sigmoid_v1, v1_mean = self.propdown(h0_sample)
        v1_sample = trng.binomial(size=v1_mean.shape, n=1, p=v1_mean, dtype=theano.config.floatX)
        return [pre_sigmoid_v1, v1_mean, v1_sample]

    def gibbs_hvh(self, h0_sample):
        pre_sigmoid_v1, v1_mean, v1_sample = self.sample_v_given_h(h0_sample)
        pre_sigmoid_h1, h1_mean, h1_sample = self.sample_h_given_v(v1_sample)
        return [pre_sigmoid_v1, v1_mean, v1_sample,
                pre_sigmoid_h1, h1_mean, h1_sample]

    def gibbs_vhv(self, v0_sample):
        pre_sigmoid_h1, h1_mean, h1_sample = self.sample_h_given_v(v0_sample)
        pre_sigmoid_v1, v1_mean, v1_sample = self.sample_v_given_h(h1_sample)
        return [pre_sigmoid_h1, h1_mean, h1_sample,
                pre_sigmoid_v1, v1_mean, v1_sample]

    def get_cost_updates(self, lr=0.1, persistent=None, k=1):
        pre_sigmoid_ph, ph_mean, ph_sample = self.sample_h_given_v(self.input)
        if persistent is None:
            chain_start = ph_sample
        else:
            chain_start = persistent
        (
            [
                pre_sigmoid_nvs,
                _,
                nv_samples,
                _,
                _,
                nh_samples
            ],
            updates
        ) = theano.scan(self.gibbs_hvh,
                        outputs_info=[None, None, None, None, None, chain_start],
                        n_steps=k,
                        name="gibbs_hvh")
        chain_end = nv_samples[-1]
        cost = T.mean(self.free_energy(self.input)) - T.mean(self.free_energy(chain_end))
        for param in self.params:
            gparam = T.grad(cost, param, consider_constant=[chain_end])
            updates[param] = param - gparam * T.cast(lr, dtype=theano.config.floatX)

        if persistent:
            updates[persistent] = nh_samples[-1]
            monitoring_cost = self.get_pseudo_likelihood_cost(updates)
        else:
            monitoring_cost = self.get_reconstruction_cost(pre_sigmoid_nvs[-1])

        return monitoring_cost, updates

    def get_pseudo_likelihood_cost(self, updates):
        bit_i_idx = theano.shared(value=0, name='bit_i_idx')
        xi = T.round(self.input)
        fe_xi = self.free_energy(xi)
        xi_flip = T.set_subtensor(xi[:, bit_i_idx], 1 - xi[:, bit_i_idx])
        fe_xi_flip = self.free_energy(xi_flip)
        cost = T.mean(self.n_visible * T.log(activation(fe_xi_flip - fe_xi)))
        updates[bit_i_idx] = (bit_i_idx + 1) % self.n_visible
        return cost

    def get_reconstruction_cost(self, pre_sigmoid_nv):
        cross_entropy = T.mean(T.sum(self.input * T.log(activation(pre_sigmoid_nv)) +
                                     (1 - self.input) * T.log(1 - activation(pre_sigmoid_nv)), axis=1))
        return cross_entropy

if __name__ == '__main__':

    # params
    plot_every = 1000
    n_samples = 10
    n_chains = 10
    training_epochs = 15
    batch_size = 50
    n_hidden = 500
    learning_rate = 0.1
    save_dir = 'data/mnist_images'

    # constants
    if 'MNIST_PATH' not in os.environ:
        print('Set MNIST_PATH as a path variable. Can download MNIST data from http://deeplearning.net/data/mnist/mnist.pkl.gz')
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
    x = T.matrix('x')

    persistent_chain = theano.shared(np.zeros((batch_size, n_hidden), dtype=theano.config.floatX), borrow=True)
    rbm = RBM(input=x, n_visible=28*28, n_hidden=n_hidden)
    cost, updates = rbm.get_cost_updates(lr=learning_rate, persistent=persistent_chain, k=15)

    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    os.chdir(save_dir)

    slice = index * batch_size

    train_rbm = theano.function(inputs=[index],
                                outputs=cost,
                                updates=updates,
                                givens={ x: train_X[index * batch_size : (index + 1) * batch_size] },
                                name='train_rbm')

    plotting_time = 0.
    pretrain_time = 0.
    start_time = timeit.default_timer()

    for epoch in range(training_epochs):
        start_time = timeit.default_timer()
        mean_cost = list()

        for batch_index in range(n_train_batches):
            mean_cost.append(train_rbm(batch_index))

            # progress bar
            frac = (n_train_batches - batch_index) * 10 / n_train_batches
            print('\r[' + '=' * (9 - frac) + '>' + ' ' * frac + ']', end='')

        stop_time = timeit.default_timer()
        pretrain_time += (start_time - stop_time)

        print(' :: Training epoch %d, took %f seconds, cost is' % (epoch, stop_time - start_time), np.mean(mean_cost))

        plotting_start = timeit.default_timer()
        image = Image.fromarray(tile_raster_images(X=rbm.W.get_value(borrow=True).T, img_shape=(28, 28), tile_shape=(10, 10), tile_spacing=(1, 1)))

        image.save('filters_at_epoch_%i.png' % epoch)
        plotting_stop = timeit.default_timer()
        plotting_time += (plotting_stop - plotting_start)

    print('Training took %f minutes' % (pretrain_time / 60.))
    number_of_test_samples = test_X.get_value(borrow=True).shape[0]

    test_idx = rng.randint(number_of_test_samples - n_chains)
    persistent_vis_chain = theano.shared(np.asarray(test_X.get_value(borrow=True)[test_idx:test_idx+n_chains], dtype=theano.config.floatX))

    (
        [
            _,
            _,
            _,
            _,
            vis_mfs,
            vis_samples
        ],
        updates
    ) = theano.scan(rbm.gibbs_vhv,
                    outputs_info=[None, None, None, None, None, persistent_vis_chain],
                    n_steps=plot_every,
                    name='gibbs_vhv')

    updates.update({ persistent_vis_chain: vis_samples[-1] })

    sample_fn = theano.function(inputs=[], outputs=[vis_mfs[-1], vis_samples[-1]], updates=updates, name='sample_fn')
    image_data = np.zeros((29 * n_samples + 1, 29 * n_chains - 1), dtype='uint8')

    for idx in range(n_samples):
        vis_mf, vis_samples = sample_fn()
        print(' ... plotting sample %d' % idx)
        image_data[29*idx:29*idx+28, :] = tile_raster_images(X=vis_mf, img_shape=(28, 28), tile_shape=(1, n_chains), tile_spacing=(1, 1))

    image = Image.fromarray(image_data)
    image.save('samples.png')
