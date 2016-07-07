from __future__ import print_function

from models.rbm import RBM
from utils import init_glorot, init_zeros, sigm, trng, rng, dtype

import theano
import theano.tensor as T
import numpy as np
import math


class InvariantRBM(RBM):
    """ Still needs some work """

    def __init__(self, n_hid, n_filt):
        super(InvariantRBM, self).__init__(n_hid)
        self.n_filt = n_filt

    def free_energy(self, v_sample):
        repr = T.tensordot(v_sample, self.R, axes=[1, 0])
        hidden_term = (T.log(1 + T.exp(T.tensordot(repr, self.W, axes=[2, 0]) + self.hbias))).sum()
        vbias_term = (v_sample + self.vbias).sum()
        return -hidden_term - vbias_term

    def visible_to_hidden(self, v_sample):
        h_inner = T.tensordot(v_sample, self.R, axes=[1, 0])
        h_inner = T.tensordot(h_inner, self.W, axes=[2, 0]) + self.hbias
        mean_h = T.exp(h_inner) / (1 + T.sum(T.exp(h_inner), axis=1, keepdims=True))
        return trng.multinomial(pvals=mean_h.transpose(0, 2, 1), dtype=dtype).transpose(0, 2, 1)

    def hidden_to_visible(self, h_sample):
        rw = T.tensordot(self.R, self.W, axes=[2, 0])
        return sigm(T.tensordot(h_sample, rw, axes=[[1, 2], [1, 2]]) + self.vbias)

    def build(self, batch_size, n_vis):
        self.batch_size = batch_size
        self.n_vis = n_vis
        self.n_rep = n_vis

        self.W = init_glorot(name='W', shape=(self.n_rep, self.n_hid))
        self.R = init_glorot(name='R', shape=(self.n_vis, self.n_filt, self.n_rep))
        self.hbias = init_zeros(name='hbias', shape=(self.n_hid,))
        self.vbias = init_zeros(name='vbias', shape=(self.n_vis,))
        self.params = [self.W, self.hbias, self.vbias]

        self._built = True

if __name__ == '__main__':
    from utils import evaluate
    evaluate(InvariantRBM(n_hid=500, n_filt=1600), 'mnist', save_dir='mnist_invariant')
