from __future__ import print_function

import random
import sys
import os
import gzip

from scipy.ndimage import rotate, shift

from rbm_utils import rng, sigm, dtype, init_glorot, init_zeros

try:
    import cPickle as pkl
except ImportError:
    import pickle as pkl

import theano
import theano.tensor as T
import numpy as np

try:
    import PIL.Image as Image
except ImportError:
    import Image

from pylearn2.utils.image import tile_raster_images


def evaluate_picture(model, **kwargs):
    """ Currently this doesn't work very well """

    if 'MY_PICTURE' not in os.environ:
        print('Set MY_PICTURE as an environment variable (path to picture you want to train on)')
        sys.exit(1)
    picture_path = os.environ['MY_PICTURE']
    if not os.path.isfile(picture_path):
        print('Nothing found at "%s"' % picture_path)
        sys.exit(1)

    if 'IMAGE_PATH' not in os.environ:
        print('Set IMAGE_PATH as an environment variable (parent directory to save generaged images)')
        sys.exit(1)
    image_path = os.path.join(os.environ['IMAGE_PATH'], kwargs.get('save_dir', 'picture'))

    if not os.path.exists(image_path):
        os.makedirs(image_path)
    os.chdir(image_path)

    im = Image.open(picture_path).convert('L')

    height, width = im.size
    greyscale_map = im.getdata()
    greyscale_map = np.array(greyscale_map, dtype='uint8')
    data = greyscale_map.reshape(width, height)

    output = Image.fromarray(data)
    output.save('original_image.png')

    data = np.asarray(data, dtype=dtype) / np.max(data)

    n_train_batches = 500
    batch_size = 25
    nb_epochs = 15
    k = 15
    lr = 0.1
    use_momentum = True
    momentum = 0.5
    persistent = True

    train_data = np.zeros((n_train_batches, width * height), dtype='uint8')

    for i in range(n_train_batches):
        translated = shift(data, (random.randint(-3, 3), random.randint(-3, 3)), mode='wrap')
        # if random.random() > 0.8:
        #     translated = rotate(translated, 180, reshape=False, mode='wrap')
        # if random.random() > 0.8:
        #     translated = np.fliplr(translated)
        rotated = rotate(translated, random.random() * 6 - 3, reshape=False, mode='wrap')
        rotated[np.random.rand(width, height) > 0.8] = 0
        train_data[i] = rotated.reshape(1, width * height)

    for epoch in range(nb_epochs):
        model.train(train_data, nb_epochs=20, batch_size=batch_size, k=k, lr=lr, use_momentum=use_momentum,
                    momentum=momentum, persistent=persistent)
        image = Image.fromarray(tile_raster_images(X=model.W.get_value(borrow=True).T, img_shape=(width, height),
                                                   tile_shape=(5, 5), tile_spacing=(1, 1)))
        image.save('filters_at_epoch_%i.png' % epoch)
        lr *= 0.9
        print('Completed metaepoch %d, learning rate is now %.3e' % (epoch, lr))

        if epoch > nb_epochs / 2:
            momentum = 0.9

    n_chains = 1
    n_samples = 10
    n_steps = 10

    data = data.reshape(1, width * height)

    sample_function = model.get_sampler(data)
    image_data = np.zeros(((width + 1) * n_samples + 1, (height + 1) * n_chains - 1), dtype='uint8')

    for idx in range(n_samples):
        image_data[(width + 1) * idx: (width + 1) * idx + width, :] = tile_raster_images(X=data, img_shape=(width, height),
                                                                                         tile_shape=(1, n_chains), tile_spacing=(1, 1))
        data = sample_function(n_steps)

    image = Image.fromarray(image_data)
    image.save('samples.png')

def evaluate_mnist(model, **kwargs):
    if 'MNIST_PATH' not in os.environ:
        print('You must set MNIST_PATH as an environment variable (pointing at mnist.pkl.gz). You can download the ' +
              'MNIST data from http://deeplearning.net/data/mnist/mnist.pkl.gz')
        sys.exit(1)
    mnist_path = os.environ['MNIST_PATH']

    if 'IMAGE_PATH' not in os.environ:
        print('Set IMAGE_PATH as an environment variable (parent directory to save generaged images)')
        sys.exit(1)
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
    nb_epochs = 50
    k = 1
    lr = 0.1
    use_momentum = True
    momentum = 0.5
    persistent = True

    for epoch in range(nb_epochs):
        model.train(train_X, nb_epochs=1, batch_size=batch_size, k=k, lr=lr, use_momentum=use_momentum,
                    momentum=momentum, validation_data=validation_X[0:256], persistent=persistent)
        image = Image.fromarray(tile_raster_images(X=model.W.get_value(borrow=True).T, img_shape=(28, 28),
                                                   tile_shape=(10, 10), tile_spacing=(1, 1)))
        image.save('filters_at_epoch_%i.png' % epoch)

        if epoch == 20:
            momentum = 0.9

        k = epoch / 5 + 1
        lr *= 0.95

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

    # a prediction model (needs improvement)
    model.build(batch_size, test_X.shape[1])

    input = T.matrix('input', dtype=dtype)
    target = T.matrix('target', dtype=dtype)
    target_idx = T.vector('target_idx', dtype='int32')
    W2 = init_glorot('W2', shape=(model.n_hid, 10))
    b2 = init_zeros('b2', shape=(10,))
    z1 = model.visible_to_hidden(input)
    z2 = T.nnet.softmax(T.dot(z1, W2) + b2)
    params = [W2, b2]
    cost = ((target - z2) ** 2).mean()
    eval_cost = T.cast(T.eq(T.argmax(z2, axis=1), target_idx), dtype='int32').mean()
    updates = [(p, p - 0.1 * T.grad(cost, p)) for p in params]
    train = theano.function(inputs=[input, target], outputs=[], updates=updates)
    test = theano.function(inputs=[input, target_idx], outputs=eval_cost)

    n_batches = train_X.shape[0] // batch_size
    nb_epochs = 50

    for epoch in range(nb_epochs):
        for batch in range(n_batches):
            data_X = train_X[batch * batch_size: (batch + 1) * batch_size]
            idx_y = train_y[batch * batch_size: (batch + 1) * batch_size]
            data_y = np.zeros(shape=(idx_y.shape[0], 10), dtype=dtype)
            data_y[np.arange(idx_y.shape[0]), idx_y] = 1
            train(data_X, data_y)
        print('Epoch %d: %.3f' % (epoch, test(test_X, test_y)))

evaluation_methods = {
    'mnist': evaluate_mnist,
    'picture': evaluate_picture,
}


def evaluate(model, dataset, **kwargs):
    dataset = dataset.lower()
    if dataset not in evaluation_methods:
        print('Dataset "%s" not available. Available: [ %s ]' % (dataset, ', '.join(evaluation_methods.keys())))
        sys.exit(1)
    message = '| Evaluating model on "%s" with settings: %s |' % (dataset, str(kwargs))
    print('o' + '-' * (len(message) - 2) + 'o\n' + message + '\no' + '-' * (len(message) - 2) + 'o')
    evaluation_methods[dataset](model, **kwargs)
