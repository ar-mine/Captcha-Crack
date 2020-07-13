from mxnet import autograd, contrib, gluon, image, init, np, npx
from mxnet.gluon import nn

npx.set_np()


def cls_predictor(num_anchors, num_classes):
    return nn.Conv2D(num_anchors * (num_classes + 1), kernel_size=3, padding=1)


print("test")