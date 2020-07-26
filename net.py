import mxnet
from mxnet.gluon import nn

class CNN:
    def __init__(self, num_classes):
        self._num_classes = num_classes

    def _sub_block(self, channel):
        net = nn.HybridSequential()
        net.add(nn.Conv2D(channel, kernel_size=3, strides=1, padding=1, activation='relu'),
                nn.BatchNorm(),
                nn.Dropout(0.5),
                nn.MaxPool2D(2))
        return net

    def get_net(self):
        net = nn.HybridSequential()
        for n in (32, 64, 128):
            net.add(self._sub_block(n))
        net.add(nn.Dense(1024),
                nn.Dense(self._num_classes*4))
        return net
