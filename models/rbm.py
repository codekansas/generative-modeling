from __future__ import print_function

import theano
import theano.tensor as T
from theano.compat.python2x import OrderedDict

from utils import init_zeros, init_glorot, dtype, sigm, binomial

import datetime


class RBM(object):
    def __init__(self, n_hid):
        self.input = T.matrix('input', dtype=dtype)
        self.n_hid = n_hid
        self._built = False

    def free_energy(self, v_sample):
        wx_b = T.dot(v_sample, self.W) + self.hbias
        vbias_term = T.dot(v_sample, self.vbias)
        hidden_term = T.sum(T.log(1 + T.exp(wx_b)), axis=1)
        return -hidden_term - vbias_term

    def visible_to_hidden(self, v_sample):
        wx_b = T.dot(v_sample, self.W) + self.hbias
        return binomial(sigm(wx_b))

    def hidden_to_visible(self, h_sample):
        wx_b = T.dot(h_sample, self.W.T) + self.vbias
        return sigm(wx_b)

    def gibbs_step_hidden(self, h_sample):
        v_sample = self.hidden_to_visible(h_sample)
        h_new = self.visible_to_hidden(v_sample)
        return v_sample, h_new

    def gibbs_step_visible(self, v_sample):
        h_sample = self.visible_to_hidden(v_sample)
        v_new = self.hidden_to_visible(h_sample)
        return v_new

    def build(self, batch_size, n_vis):
        self.batch_size = batch_size
        self.n_vis = n_vis

        self.W = init_glorot(name='W', shape=(self.n_vis, self.n_hid))
        self.hbias = init_zeros(name='hbias', shape=(self.n_hid,), offset=-4)
        self.vbias = init_zeros(name='vbias', shape=(self.n_vis,), offset=-4)
        self.params = [self.W, self.hbias, self.vbias]

        self._built = True

    def calculate_updates(self, k, lr, use_momentum=False, momentum=None, persistent=False):
        if persistent:
            persistent_chain = init_zeros('persistent_chain', shape=(self.batch_size, self.n_hid))
            chain_start = persistent_chain
        else:
            chain_start = self.visible_to_hidden(self.input)

        ([vs, hs], updates) = theano.scan(self.gibbs_step_hidden,
                                          outputs_info=[None, chain_start],
                                          n_steps=k)
        chain_end = vs[-1]
        cost = T.mean(self.free_energy(self.input)) - T.mean(self.free_energy(chain_end))

        if persistent:
            updates[persistent_chain] = hs[-1]
        updates.update(self.get_gradients(cost, chain_end, lr, use_momentum, momentum))

        return updates

    def get_gradients(self, cost, chain_end, lr, use_momentum=False, momentum=None):
        updates = OrderedDict()
        if use_momentum:
            assert momentum is not None
            for param in self.params:
                gradient = T.grad(cost, param, consider_constant=[chain_end])
                velocity = init_zeros('velocity_' + str(param.name), shape=param.get_value(borrow=True).shape)
                update = param - T.cast(lr, dtype=dtype) * gradient
                x = momentum * velocity + update - param
                updates[velocity] = x
                updates[param] = momentum * x + update
        else:
            for param in self.params:
                gradient = T.grad(cost, param, consider_constant=[chain_end])
                updates[param] = param - T.cast(lr, dtype=dtype) * gradient + momentum
        return updates

    def train(self, data, nb_epochs=1, batch_size=10, k=1, lr=0.1, persistent=False,
              use_momentum=False, momentum=None, validation_data=None):
        if not self._built:
            self.build(batch_size=batch_size, n_vis=data.shape[1])
        assert data.ndim == 2 and self.batch_size == batch_size and self.n_vis == data.shape[1]

        if not use_momentum:
            momentum = 0.

        if not hasattr(self, 'train_func'):
            # doing this allows us to change hyperparameters after creating the training function
            k_var = T.lscalar('k')
            lr_var = T.scalar('lr', dtype=dtype)
            mu_var = T.scalar('mu', dtype=dtype)
            updates = self.calculate_updates(k=k_var, lr=lr_var, momentum=mu_var,
                                             use_momentum=use_momentum, persistent=persistent)
            self.train_func = theano.function(inputs=[self.input, k_var, lr_var, mu_var],
                                              outputs=[],
                                              updates=updates,
                                              name='training_function')

        n_train_batches = data.shape[0] // batch_size
        n_grades = 10

        cost_monitor = self.get_cost_monitor()

        for epoch in range(1, nb_epochs + 1):
            start_time = datetime.datetime.now()

            for batch in range(n_train_batches):
                batch_data = data[batch * batch_size: (batch + 1) * batch_size]
                self.train_func(batch_data, k, lr, momentum)
                fraction = (n_train_batches - batch) * n_grades / n_train_batches

                messages = list()
                messages.append('[' + '=' * (n_grades - fraction - 1) + '>' + ' ' * fraction + ']')
                messages.append('Epoch: %d / %d' % (epoch, nb_epochs))
                messages.append('Time: %s' % str(datetime.datetime.now() - start_time))
                # messages.append('Cost: %.3e' % -cost_monitor(batch_data))
                messages.append('(%d / %d)' % (batch, n_train_batches))
                print('\r' + ' '.join(messages), end='')

            messages = list()
            messages.append('[' + '=' * n_grades + ']')
            messages.append('Epoch: %d / %d' % (epoch, nb_epochs))
            messages.append('Time: %s' % str(datetime.datetime.now() - start_time))

            if validation_data is not None:
                assert validation_data.ndim == 2 and validation_data.shape[1] == self.n_vis
                cost = -cost_monitor(validation_data)
                messages.append('Validation cost: %.3e' % cost.mean())

            print('\r' + ' '.join(messages))

    def get_sampler(self, data):
        if not self._built:
            self.build(batch_size=data.shape[0], n_vis=data.shape[1])
        assert data.ndim == 2 and self.n_vis == data.shape[1]

        persistent_vis_chain = theano.shared(data)

        n_steps = T.lscalar('n_steps')
        v_sample, updates = theano.scan(self.gibbs_step_visible,
                                        outputs_info=[persistent_vis_chain],
                                        n_steps=n_steps)
        updates[persistent_vis_chain] = v_sample[-1]
        sample_function = theano.function(inputs=[n_steps],
                                          outputs=v_sample[-1],
                                          updates=updates,
                                          name='sample_function')
        return sample_function

    def get_cost_monitor(self):
        if not hasattr(self, 'cost_monitor'):
            bit_i_idx = theano.shared(value=0, name='bit_i_idx')
            xi = T.round(self.input)
            fe_xi = self.free_energy(xi)
            xi_flip = T.set_subtensor(xi[:, bit_i_idx], 1 - xi[:, bit_i_idx])
            fe_xi_flip = self.free_energy(xi_flip)
            cost = T.mean(self.n_vis * T.log(sigm(fe_xi_flip - fe_xi)))
            self.cost_monitor = theano.function(inputs=[self.input],
                                                outputs=cost,
                                                updates={bit_i_idx: (bit_i_idx + 1) % self.n_vis},
                                                name='monitoring_function')
        return self.cost_monitor

if __name__ == '__main__':
    from utils import evaluate
    evaluate(RBM(n_hid=500), 'mnist', save_dir='picture_vanilla')
