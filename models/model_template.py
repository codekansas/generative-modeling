
import theano
import theano.tensor as T
import numpy as np

from utils.rbm_utils import init_zeros, dtype


class AbstractModel():
    def __init__(self, n_vis, n_hid):
        self.n_vis = n_vis
        self.n_hid = n_hid

        self.lr = T.dscalar('learning_rate')
        self.k = T.lscalar('cd_steps')

        self.updates = dict()

    def get_monitoring_cost(self):
        return 1

    def get_training_function(self):
        func = theano.function(inputs=[self.lr, self.k],
                               outputs=self.get_monitoring_cost(),
                               updates=self.updates,
                               name='training_function')
        return func

    def gibbs_step(self):
        pass

    def get_sample_function(self, plot_every=10, n_chains=1):
        persistent_vis_chain = init_zeros(name='persistent', shape=(n_chains,self.n_vis))
        ([vis_mfs, vis_samples], updates) = theano.scan(self.gibbs_step(),
                                                        outputs_info=[None, persistent_vis_chain],
                                                        n_steps=plot_every,
                                                        name='gibbs_sampling')