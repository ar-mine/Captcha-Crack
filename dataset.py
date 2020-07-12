from mxnet import gluon, init, nd
from mxnet.gluon import data as gdata, nn
import os
import sys


class Dataset:
    def __init__(self, data_name, batch_size, root='.\\asset\\dataset', resize=None):
        if data_name == 1:
            path = os.path.join(root, 'fashion_mnist')
            self.train_iter, self.test_iter = load_data_fashion_mnist(batch_size, path, resize)


def load_data_fashion_mnist(batch_size, path, resize=None):
    transformer = []
    if resize:
        transformer += [gdata.vision.transforms.Resize(resize)]
    transformer += [gdata.vision.transforms.ToTensor()]
    transformer = gdata.vision.transforms.Compose(transformer)
    mnist_train = gdata.vision.FashionMNIST(root=path, train=True)
    mnist_test = gdata.vision.FashionMNIST(root=path, train=False)
    num_workers = 0 if sys.platform.startswith('win32') else 4
    train_iter = gdata.DataLoader(
        mnist_train.transform_first(transformer), batch_size, shuffle=True,
        num_workers=num_workers)
    test_iter = gdata.DataLoader(
        mnist_test.transform_first(transformer), batch_size, shuffle=False,
        num_workers=num_workers)
    return train_iter, test_iter

