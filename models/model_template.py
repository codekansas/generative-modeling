
import theano
import theano.tensor as T
import numpy as np

from utils.rbm_utils import init_zeros, init_glorot, dtype, sigm, binomial


class AbstractModel():
    def __init__(self, n_vis, n_hid):
        self.n_vis = n_vis
        self.n_hid = n_hid

        self.input = T.vector('input', dtype=dtype)
        self.lr = T.dscalar('learning_rate')
        self.k = T.lscalar('cd_steps')
        self.updates = dict()
        self.params = list()

        self._built = False
        self.init_params()

    def init_params(self):
        self.W = init_glorot(name='W', shape=(self.n_vis, self.n_hid))
        self.hbias = init_zeros(name='hbias', shape=(self.n_hid,))
        self.vbias = init_zeros(name='vbias', shape=(self.n_vis,))

        self.params += [self.W, self.hbias, self.vbias]

    def free_energy(self, v_sample):
        wx_b = T.dot(v_sample, self.W) + self.hbias
        vbias_term = T.dot(v_sample, self.vbias)
        hidden_term = T.sum(T.log(1 + T.exp(wx_b)), axis=1)
        return -(hidden_term + vbias_term)

    def visible_to_hidden(self, v_sample):
        wx_b = T.dot(v_sample, self.W) + self.hbias
        return sigm(wx_b)

    def hidden_to_visible(self, h_sample):
        wx_b = T.dot(h_sample, self.W.T) + self.vbias
        return sigm(wx_b)

    def build(self, n_batches):
        persistent_chain = init_zeros('persistent_chain', shape=(n_batches, self.n_hid))

        def gibbs_step_hidden(h_sample):
            v_sample = self.hidden_to_visible(h_sample)
            h_new = binomial(self.visible_to_hidden(v_sample))
            return v_sample, h_new

        ([vs, hs], updates) = theano.scan(gibbs_step_hidden, outputs_info=[None, persistent_chain], n_steps=self.k)
        chain_end = vs[-1]

        cost = T.mean(self.free_energy(self.input) - T.mean(self.free_energy(chain_end)))
        for param in self.params:
            gradient = T.grad(cost, param, consider_constant=[chain_end])
            updates[param] = param - gradient * self.lr

        updates[persistent_chain] = hs[-1]
        self._built = True

    def get_monitoring_cost(self):
        bit_i_idx = theano.shared(value=0, name='bit_i_idx')
        xi = T.round(self.input)
        fe_xi = self.free_energy(xi)
        xi_flip = T.set_subtensor(xi[:, bit_i_idx], 1 - xi[:, bit_i_idx])
        fe_xi_flip = self.free_energy(xi_flip)
        cost = T.mean(self.n_vis * T.log(sigm(fe_xi_flip - fe_xi)))
        self.updates[bit_i_idx] = (bit_i_idx + 1) % self.n_vis
        return cost

    def get_training_function(self, n_batches, monitor_cost=False):
        if not self._built:
            self.build(n_batches)

        train_func = theano.function(inputs=[self.input, self.lr, self.k],
                                     outputs=self.get_monitoring_cost() if monitor_cost else None,
                                     updates=self.updates,
                                     name='train_function')

        return train_func

if __name__ == '__main__':
    print('Testing')