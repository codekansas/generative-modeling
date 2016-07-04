import numpy as np

import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

rng = np.random.RandomState(42)
trng = RandomStreams(rng.randint(2 ** 30))
sigm = lambda x: T.nnet.sigmoid(x)
dtype = theano.config.floatX


def init_zeros(name, shape):
    value = np.zeros(shape=shape, dtype=theano.config.floatX)
    return theano.shared(name=name, value=value, borrow=True)

def glorot_init(name, shape):
    magnitude = 4 * np.sqrt(6. / sum(shape))
    value = np.asarray(rng.uniform(low=-magnitude, high=magnitude, size=shape), dtype=dtype)
    return theano.shared(name=name, value=value, borrow=True)
