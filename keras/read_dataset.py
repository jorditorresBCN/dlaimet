from __future__ import absolute_import
from keras.datasets.cifar import load_batch
from keras.utils.data_utils import get_file
from keras import backend as K
import numpy as np
import os


def load_data(dirname):
    """Loads CIFAR10 dataset locally.

    # Returns
        Tuple of Numpy arrays: `(x_train, y_train), (x_test, y_test)`.
    """
    try:
        if dirname is not None:
            path = os.path.abspath(dirname)
        else:
            dirname_remote = 'cifar-10-batches-py'
            origin = 'http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
            path = get_file(dirname_remote, origin=origin, untar=True)

        num_train_samples = 50000

        x_train = np.zeros((num_train_samples, 3, 32, 32), dtype='uint8')
        y_train = np.zeros((num_train_samples,), dtype='uint8')

        for i in range(1, 6):
            fpath = os.path.join(path, 'data_batch_' + str(i))
            data, labels = load_batch(fpath)
            x_train[(i - 1) * 10000: i * 10000, :, :, :] = data
            y_train[(i - 1) * 10000: i * 10000] = labels

        fpath = os.path.join(path, 'test_batch')
        x_test, y_test = load_batch(fpath)

        y_train = np.reshape(y_train, (len(y_train), 1))
        y_test = np.reshape(y_test, (len(y_test), 1))

        if K.image_data_format() == 'channels_last':
            x_train = x_train.transpose(0, 2, 3, 1)
            x_test = x_test.transpose(0, 2, 3, 1)

        return (x_train, y_train), (x_test, y_test)

    except FileNotFoundError as err:
        print(
            "ERROR: THERE AREN'T LOCAL FILES, IF YOU WANT TO DOWNLOAD THE DATASET, SET dirname TO None. \n {0}".format(
                err))
