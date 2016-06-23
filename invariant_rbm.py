from __future__ import print_function

from rbm import RBM, evaluate_rbm_mnist

import theano.tensor as T


class TIRBM(object, RBM):
    def __init__(self, *args):
        super(TIRBM, self).__init__(*args)

    def free_energy(self, v_sample):
        wx_b = T.dot(v_sample, self.W) + self.hbias
        vbias_term = T.dot(v_sample, self.vbias)
        hidden_term = T.sum(T.log(1 + T.exp(wx_b)), axis=1)
        return -hidden_term - vbias_term

if __name__ == '__main__':
    evaluate_rbm_mnist(rbm_model=TIRBM, save_dir='tirbm_mnist')
