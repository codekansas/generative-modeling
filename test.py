""" theano two-layer network xor example, usually a good sanity check """

from __future__ import print_function

import theano
import theano.tensor as T
import numpy as np

X = T.matrix(name='X', dtype=theano.config.floatX)
y = T.matrix(name='y', dtype=theano.config.floatX)
lr = T.scalar(name='lr', dtype=theano.config.floatX)

n_in, n_hid, n_out = 2, 3, 1

def get_shared(*shape):
    return theano.shared(np.random.rand(*shape), name='{}_weights'.format(shape), strict=False)

W1, b1 = get_shared(n_in, n_hid), get_shared(n_hid)
a1 = T.dot(X, W1) + b1
z1 = T.tanh(a1)

W2, b2 = get_shared(n_hid, n_out), get_shared(n_out)
a2 = T.dot(z1, W2) + b2
z2 = T.tanh(a2)

cost = 0.5 * ((z2 - y) ** 2).sum()

updates = [(p, p - lr * T.grad(cost, p)) for p in [W1, b1, W2, b2]]
train = theano.function([X, y, lr], [cost], updates=updates)
test = theano.function([X], [z2])

X_data = np.asarray([[0, 0], [1, 1], [0, 1], [1, 0]])
y_data = np.asarray([[0], [0], [1], [1]])

print('Training...')
for i in range(10000):
    train(X_data, y_data, 0.01)
    if i % 1000 == 0:
        print('at iteration %d:' % i, test(X_data)[0])

print('Testing...')
print(test(X_data))
