""" theano two-layer network xor example, usually a good sanity check """

from __future__ import print_function

import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
import numpy as np

# X = T.matrix(name='X', dtype=theano.config.floatX)
# y = T.matrix(name='y', dtype=theano.config.floatX)
# lr = T.scalar(name='lr', dtype=theano.config.floatX)
#
# n_in, n_hid, n_out = 2, 3, 1
#
# def get_shared(*shape):
#     return theano.shared(np.random.rand(*shape), name='{}_weights'.format(shape), strict=False)
#
# W1, b1 = get_shared(n_in, n_hid), get_shared(n_hid)
# a1 = T.dot(X, W1) + b1
# z1 = T.tanh(a1)
#
# W2, b2 = get_shared(n_hid, n_out), get_shared(n_out)
# a2 = T.dot(z1, W2) + b2
# z2 = T.tanh(a2)
#
# cost = 0.5 * ((z2 - y) ** 2).sum()
#
# updates = [(p, p - lr * T.grad(cost, p)) for p in [W1, b1, W2, b2]]
# train = theano.function([X, y, lr], [cost], updates=updates)
# test = theano.function([X], [z2])
#
# X_data = np.asarray([[0, 0], [1, 1], [0, 1], [1, 0]])
# y_data = np.asarray([[0], [0], [1], [1]])
#
# print('Training...')
# for i in range(10000):
#     train(X_data, y_data, 0.01)
#     if i % 1000 == 0:
#         print('at iteration %d:' % i, test(X_data)[0])
#
# print('Testing...')
# print(test(X_data))

# rng = np.random.RandomState(42)
# trng = RandomStreams(rng.randint(2**30))
#
#
# def build_rbm(v, W, bv, bh, k):
#     def gibbs_step(v):
#         mean_h = T.nnet.sigmoid(T.dot(v, W) + bh)
#         h = trng.binomial(size=mean_h.shape, n=1, p=mean_h, dtype=theano.config.floatX)
#         mean_v = T.nnet.sigmoid(T.dot(h, W.T) + bv)
#         v = trng.binomial(size=mean_v.shape, n=1, p=mean_v, dtype=theano.config.floatX)
#         return mean_v, v
#
#     chain, updates = theano.scan(lambda v: gibbs_step(v)[1], outputs_info=[v], n_steps=k)
#     v_sample = chain[-1]
#
#     # instead of accuracy, monitor how much the gibbs step diverges
#     # lower is better (stronger probability distribution)
#     mean_v = gibbs_step(v_sample)[0]
#     monitor = T.xlogx.xlogy0(v, mean_v) + T.xlogx.xlogy0(1 - v, 1 - mean_v)
#     monitor = monitor.sum() / v.shape[0]
#
#     def free_energy(v):
#         return -(v * bv).sum() - T.log(1 + T.exp(T.dot(v, W) + bh)).sum()
#
#     cost = (free_energy(v) - free_energy(v_sample)) / v.shape[0]
#     return v_sample, cost, monitor, updates

import numpy as np

n_vis = 3
n_hid = 4
n_rep = 5
n_filt = 7
b_size = 11

rnd = lambda *shape: np.random.uniform(size=shape)

v = rnd(b_size, n_vis)
R = rnd(n_vis, n_filt, n_rep)
W = rnd(n_rep, n_hid)
vbias = rnd(n_vis)
hbias = rnd(n_filt, n_hid)

repr = np.tensordot(v, R, axes=[1, 0])
out = np.tensordot(repr, W, axes=[2, 0]) + hbias
vterm = (v * vbias).sum()
print(vterm.shape)