from __future__ import print_function

import sys
import os
import gzip

from utils.rbm_utils import rng

try:
    import cPickle as pkl
except ImportError:
    import pickle as pkl

import theano
import numpy as np

try:
    import PIL.Image as Image
except ImportError:
    import Image

from pylearn2.utils.image import tile_raster_images


def evaluate_mnist(model, **kwargs):
    if 'MNIST_PATH' not in os.environ:
        print('You must set MNIST_PATH as an environment variable (pointing at mnist.pkl.gz). You can download the ' +
              'MNIST data from http://deeplearning.net/data/mnist/mnist.pkl.gz')
        sys.exit(1)
    mnist_path = os.environ['MNIST_PATH']

    if 'IMAGE_PATH' not in os.environ:
        print('Set IMAGE_PATH as an environment variable (parent directory to save generaged images)')
    image_path = os.path.join(os.environ['IMAGE_PATH'], kwargs.get('save_dir', 'mnist'))

    print('Loading MNIST files from "%s"' % mnist_path)
    print('Saving files in "%s"' % image_path)

    if not os.path.exists(image_path):
        os.makedirs(image_path)
    os.chdir(image_path)

    f = gzip.open(mnist_path, 'rb')
    train_set, validation_set, test_set = pkl.load(f)

    def split_data(xy):
        x, y = xy
        x = np.asarray(x, dtype=theano.config.floatX)
        y = np.asarray(y, dtype='int32')
        return x, y

    train_X, train_y = split_data(train_set)
    validation_X, validation_y = split_data(validation_set)
    test_X, test_y = split_data(test_set)

    # hyperparameters
    batch_size = 20
    nb_epochs = 15
    k = 15
    lr = 0.1
    use_momentum = True
    momentum = 0.9
    persistent = True

    for epoch in range(nb_epochs):
        model.train(train_X, nb_epochs=1, batch_size=batch_size, k=k, lr=lr, use_momentum=use_momentum,
                    momentum=momentum, validation_data=validation_X[0:256], persistent=persistent)
        image = Image.fromarray(tile_raster_images(X=model.W.get_value(borrow=True).T, img_shape=(28, 28),
                                                   tile_shape=(10, 10), tile_spacing=(1, 1)))
        image.save('filters_at_epoch_%i.png' % epoch)

    # testing params
    number_of_test_samples = test_X.shape[0]
    n_chains = 10
    n_samples = 10
    n_steps = 1000

    test_idx = rng.randint(number_of_test_samples - n_chains)
    v_sample = test_X[test_idx:test_idx+n_chains]

    sample_function = model.get_sampler(v_sample)
    image_data = np.zeros((29 * n_samples + 1, 29 * n_chains - 1), dtype='uint8')

    for idx in range(n_samples):
        image_data[29 * idx: 29 * idx + 28, :] = tile_raster_images(X=v_sample, img_shape=(28, 28),
                                                                    tile_shape=(1, n_chains), tile_spacing=(1, 1))
        v_sample = sample_function(n_steps)

    image = Image.fromarray(image_data)
    image.save('samples.png')

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
