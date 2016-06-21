from __future__ import print_function

from rbm import RBM


class InvariantRBM(object, RBM):

    def __init__(self, n_visible, n_hidden, input):
        RBM.__init__(self, n_visible, n_hidden, input)
