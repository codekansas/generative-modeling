import sys
import os
import gzip

try:
    import cPickle as pkl
except ImportError:
    import pickle as pkl

import theano
import numpy as np
import theano.tensor as T

from rbm_utils import rng, dtype


def evaluate_mnist(model, **kwargs):
    if 'MNIST_PATH' not in os.environ:
        print('You must set MNIST_PATH as an environment variable (pointing at mnist.pkl.gz). You can download the ' +
              'MNIST data from http://deeplearning.net/data/mnist/mnist.pkl.gz')
        sys.exit(1)
    mnist_path = os.environ['MNIST_PATH']

    if 'IMAGE_PATH' not in os.environ:
        print('Set IMAGE_PATH as an environment variable (parent directory to save generaged images)')
    image_path = os.path.join(os.environ['IMAGE_PATH'], kwargs.get('save_dir', 'mnist'))

    print('Loading MNIST files from [ %s ]' % mnist_path)
    print('Saving files in [ %s ]' % image_path)

    # Load parameters
    batch_size = kwargs.get('batch_size', 20)
    training_epochs = kwargs.get('training_epochs', 15)
    n_chains = kwargs.get('n_chains', 10)
    plot_every = kwargs.get('plot_every', 10)
    n_samples = kwargs.get('n_samples', 1)

    if not os.path.exists(image_path):
        os.makedirs(image_path)
    os.chdir(image_path)

    f = gzip.open(mnist_path, 'rb')
    train_set, _, test_set = pkl.load(f)

    def split_shared(xy):
        x, y = xy
        x = theano.shared(np.asarray(x, dtype=theano.config.floatX))
        y = theano.shared(np.asarray(y, dtype=theano.config.floatX))
        return x, T.cast(y, 'int32')

    train_X, train_y = split_shared(train_set)
    test_X, test_y = split_shared(test_set)

    n_train_batches = train_X.get_value(borrow=True).shape[0] // batch_size
    index = T.lscalar('index')

evaluation_methods = {
    'mnist': evaluate_mnist,
}

def evaluate(model, dataset, **kwargs):
    dataset = dataset.lower()
    if dataset not in evaluation_methods:
        print('Dataset "%s" not available. Available: [ %s ]' % (dataset, ', '.join(evaluation_methods.keys())))
        sys.exit(1)
    message = '| Evaluating model on "%s" with settings: %s |' % (dataset, str(kwargs))
    print('o' + '-' * (len(message) - 2) + 'o\n' + message + '\no' + '-' * (len(message) - 2) + 'o')
    evaluation_methods[dataset](model, **kwargs)

if __name__ == '__main__':
    evaluate('a', 'mnist')
