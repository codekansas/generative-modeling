import numpy as np

import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

rng = np.random.RandomState(42)
trng = RandomStreams(rng.randint(2 ** 30))
dtype = theano.config.floatX
sigm = lambda x: T.nnet.sigmoid(x)
binomial = lambda x: trng.binomial(size=x.shape, n=1, p=x, dtype=dtype)


def init_zeros(name, shape):
    value = np.zeros(shape=shape, dtype=theano.config.floatX)
    return theano.shared(name=name, value=value, borrow=True, strict=False)


def init_glorot(name, shape):
    magnitude = 4 * np.sqrt(6. / sum(shape))
    value = np.asarray(rng.uniform(low=-magnitude, high=magnitude, size=shape), dtype=dtype)
    return theano.shared(name=name, value=value, borrow=True, strict=False)
