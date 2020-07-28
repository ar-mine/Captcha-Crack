import mxnet
from mxnet.gluon import nn


class CNN():
    def __init__(self, num_classes, c_len):
        self._num_classes = num_classes
        self._c_len = c_len

    def _sub_block(self, channel):
        net = nn.Sequential()
        net.add(nn.Conv2D(channel, kernel_size=3, strides=1, padding=1, activation='relu'),
                nn.BatchNorm(),
                nn.MaxPool2D(2))
        return net

    def get_net(self):
        net = nn.Sequential()
        for n in (32, 64, 128, 256, 256):
            net.add(self._sub_block(n))
        net.add(nn.Dense(1024),
                MultiClasses(self._num_classes, self._c_len))
        return net

class MultiClasses(nn.Block):
    def __init__(self, num_classes, c_len):
        super(MultiClasses, self).__init__()
        self._num_classes = num_classes
        self._c_len = c_len
        self.c1 = nn.Dense(self._num_classes)
        self.c2 = nn.Dense(self._num_classes)
        self.c3 = nn.Dense(self._num_classes)
        self.c4 = nn.Dense(self._num_classes)

    def forward(self, x):
        return [self.c1(x), self.c2(x), self.c3(x), self.c4(x)]
