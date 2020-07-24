# from gluoncv import model_zoo, data, utils
# from matplotlib import pyplot as plt
#
# net = model_zoo.get_model('ssd_512_resnet50_v1_voc', pretrained=True)
#
# im_fname = utils.download('https://github.com/dmlc/web-data/blob/master/' +
#                           'gluoncv/detection/street_small.jpg?raw=true',
#                           path='street_small.jpg')
# x, img = data.transforms.presets.ssd.load_test(im_fname, short=512)
# print('Shape of pre-processed image:', x.shape)
#
# class_IDs, scores, bounding_boxes = net(x)
# ax = utils.viz.plot_bbox(img, bounding_boxes[0], scores[0], class_IDs[0], class_names=net.classes)
# plt.show()

from gluoncv.data import VOCDetection
from matplotlib import pyplot as plt
from gluoncv.utils import viz
from gluoncv.data.transforms import presets
from gluoncv import utils, data
from gluoncv.data.batchify import Tuple, Stack, Pad
from mxnet.gluon.data import DataLoader
from gluoncv import model_zoo
import mxnet as mx
from mxnet import gluon, autograd, init
from gluoncv.loss import SSDMultiBoxLoss


train_dataset = VOCDetection(splits=[(2007, 'trainval'), (2012, 'trainval')], root="E:/Dataset/VOCdevkit")
val_dataset = VOCDetection(splits=[(2007, 'test')], root="E:/Dataset/VOCdevkit")


# train_image2, train_label2 = train_transform(train_image, train_label)
# print('tensor shape:', train_image2.shape)
#
# train_image2 = train_image2.transpose(
#     (1, 2, 0)) * nd.array((0.229, 0.224, 0.225)) + nd.array((0.485, 0.456, 0.406))
# train_image2 = (train_image2 * 255).clip(0, 255)
# ax = viz.plot_bbox(train_image2.asnumpy(), train_label2[:, :4],
#                    labels=train_label2[:, 4:5],
#                    class_names=train_dataset.classes)
# plt.show()

# val_image, val_label = val_dataset[0]
# bboxes = val_label[:, :4]
# cids = val_label[:, 4:5]
# ax = viz.plot_bbox(
#     val_image.asnumpy(),
#     bboxes,
#     labels=cids,
#     class_names=train_dataset.classes)
# plt.show()
#
# val_image2, val_label2 = val_transform(val_image, val_label)
# val_image2 = val_image2.transpose(
#     (1, 2, 0)) * nd.array((0.229, 0.224, 0.225)) + nd.array((0.485, 0.456, 0.406))
# val_image2 = (val_image2 * 255).clip(0, 255)
# ax = viz.plot_bbox(val_image2.clip(0, 255).asnumpy(), val_label2[:, :4],
#                    labels=val_label2[:, 4:5],
#                    class_names=train_dataset.classes)
# plt.show()


batch_size = 16
num_workers = 0
net = model_zoo.get_model('ssd_300_vgg16_atrous_voc', pretrained_base=False)


x = mx.nd.zeros(shape=(1, 3, 512, 512))
net.initialize(init=init.Xavier())
cids, scores, bboxes = net(x)
with autograd.train_mode():
    cls_preds, box_preds, anchors = net(x)
width, height = 512, 512  # suppose we use 512 as base training size
train_transform = presets.ssd.SSDDefaultTrainTransform(width, height, anchors)
val_transform = presets.ssd.SSDDefaultValTransform(width, height)


batchify_fn = Tuple(Stack(), Stack(), Stack())
train_loader = DataLoader(
    train_dataset.transform(train_transform),
    batch_size,
    shuffle=True,
    batchify_fn=batchify_fn,
    last_batch='rollover',
    num_workers=num_workers)

mbox_loss = SSDMultiBoxLoss()
trainer = gluon.Trainer(
    net.collect_params(), 'sgd',
    {'learning_rate': 0.001, 'wd': 0.0005, 'momentum': 0.9})


print("start===")
for batch in train_loader:
    with autograd.record():
        X = batch[0]
        box = batch[1]
        anc = batch[2]
        cls_pred, box_pred, anchors = net(X)
        sum_loss, cls_loss, box_loss = mbox_loss(cls_pred, box_pred, box, anc)
    # some standard gluon training steps:
    autograd.backward(sum_loss)
    trainer.step(batch_size)


im_fname = "./000005.jpg"
x, img = data.transforms.presets.ssd.load_test(im_fname, short=512)
class_IDs, scores, bounding_boxes = net(x)
ax = utils.viz.plot_bbox(img, bounding_boxes[0], scores[0], class_IDs[0], class_names=net.classes)
plt.show()
