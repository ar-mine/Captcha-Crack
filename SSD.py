from mxnet import autograd, contrib, gluon, image, init, np, npx
from mxnet.gluon import nn
import mxnet as mx
import os
import matplotlib.pyplot as plt
import time
import cv2 as cv
import armine as am


# def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5):
#     """Plot a list of images."""
#     figsize = (num_cols * scale, num_rows * scale)
#     _, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
#     axes = axes.flatten()
#     for i, (ax, img) in enumerate(zip(axes, imgs)):
#         ax.imshow(img.asnumpy())
#         ax.axes.get_xaxis().set_visible(False)
#         ax.axes.get_yaxis().set_visible(False)
#         if titles:
#             ax.set_title(titles[i])
#     return axes
#
#
# def bbox_to_rect(bbox, color):
#     """Convert bounding box to matplotlib format."""
#     # Convert the bounding box (top-left x, top-left y, bottom-right x,
#     # bottom-right y) format to matplotlib format: ((upper-left x,
#     # upper-left y), width, height)
#     return plt.Rectangle(xy=(bbox[0], bbox[1]), width=bbox[2]-bbox[0], height=bbox[3]-bbox[1],
#                         fill=False, edgecolor=color, linewidth=2)
#
#
# def show_bboxes(axes, bboxes, labels=None, colors=None):
#     """Show bounding boxes."""
#     def _make_list(obj, default_values=None):
#         if obj is None:
#             obj = default_values
#         elif not isinstance(obj, (list, tuple)):
#             obj = [obj]
#         return obj
#     labels = _make_list(labels)
#     colors = _make_list(colors, ['b', 'g', 'r', 'm', 'c'])
#     for i, bbox in enumerate(bboxes):
#         color = colors[i % len(colors)]
#         rect = bbox_to_rect(bbox.asnumpy(), color)
#         axes.add_patch(rect)
#         if labels and len(labels) > i:
#             text_color = 'k' if color == 'w' else 'w'
#             axes.text(rect.xy[0], rect.xy[1], labels[i],
#                         va='center', ha='center', fontsize=9, color=text_color,
#                         bbox=dict(facecolor=color, lw=0))


# def load_data_pikachu(batch_size, edge_size=256):
#     """Load the pikachu dataset"""
#     data_dir = './asset/dataset'
#     train_iter = image.ImageDetIter(
#         path_imgrec=os.path.join(data_dir, 'train.rec'),
#         path_imgidx=os.path.join(data_dir, 'train.idx'),
#         batch_size=batch_size,
#         data_shape=(3, edge_size, edge_size),  # The shape of the output image
#         shuffle=True,   # Read the dataset in random order
#         rand_crop=1,    # The probability of random cropping is 1
#         min_object_covered=0.95, max_attempts=200)
#     val_iter = image.ImageDetIter(
#         path_imgrec=os.path.join(data_dir, 'val.rec'),
#         batch_size=batch_size,
#         data_shape=(3, edge_size, edge_size),
#         shuffle=False)
#     return train_iter, val_iter


def cls_predictor(num_anchors, num_classes):
    return nn.Conv2D(num_anchors * (num_classes + 1), kernel_size=3, padding=1)

def bbox_predictor(num_anchors):
    return nn.Conv2D(num_anchors * 4, kernel_size=3, padding=1)

def flatten_pred(pred):
    return npx.batch_flatten(pred.transpose(0, 2, 3, 1))

def concat_preds(preds):
    return np.concatenate([flatten_pred(p) for p in preds], axis=1)

def down_sample_blk(num_channels):
    blk = nn.Sequential()
    for _ in range(2):
        blk.add(nn.Conv2D(num_channels, kernel_size=3, padding=1),
                nn.BatchNorm(in_channels=num_channels),
                nn.Activation('relu'))
    blk.add(nn.MaxPool2D(2))
    return blk

def base_net():
    blk = nn.Sequential()
    for num_filters in [16, 32, 64]:
        blk.add(down_sample_blk(num_filters))
    return blk

def get_blk(i):
    if i == 0:
        blk = base_net()
    elif i == 4:
        blk = nn.GlobalMaxPool2D()
    else:
        blk = down_sample_blk(128)
    return blk

def blk_forward(X, blk, size, ratio, cls_predictor, bbox_predictor):
    Y = blk(X)
    anchors = npx.multibox_prior(Y, sizes=size, ratios=ratio)
    cls_preds = cls_predictor(Y)
    bbox_preds = bbox_predictor(Y)
    return (Y, anchors, cls_preds, bbox_preds)

class TinySSD(nn.Block):
    def __init__(self, num_classes, **kwargs):
        super(TinySSD, self).__init__(**kwargs)
        self.num_classes = num_classes
        for i in range(5):
            # The assignment statement is self.blk_i = get_blk(i)
            setattr(self, 'blk_%d' % i, get_blk(i))
            setattr(self, 'cls_%d' % i, cls_predictor(num_anchors, num_classes))
            setattr(self, 'bbox_%d' % i, bbox_predictor(num_anchors))

    def forward(self, X):
        anchors, cls_preds, bbox_preds = [None] * 5, [None] * 5, [None] * 5
        for i in range(5):
            # getattr(self, 'blk_%d' % i) accesses self.blk_i
            X, anchors[i], cls_preds[i], bbox_preds[i] = blk_forward(
            X, getattr(self, 'blk_%d' % i), sizes[i], ratios[i],
            getattr(self, 'cls_%d' % i), getattr(self, 'bbox_%d' % i))
        # In the reshape function, 0 indicates that the batch size remains
        # unchanged
        anchors = np.concatenate(anchors, axis=1)
        cls_preds = concat_preds(cls_preds)
        cls_preds = cls_preds.reshape(
        cls_preds.shape[0], -1, self.num_classes + 1)
        bbox_preds = concat_preds(bbox_preds)
        return anchors, cls_preds, bbox_preds

def calc_loss(cls_preds, cls_labels, bbox_preds, bbox_labels, bbox_masks):
    cls = cls_loss(cls_preds, cls_labels)
    bbox = bbox_loss(bbox_preds * bbox_masks, bbox_labels * bbox_masks)
    return cls + bbox

def cls_eval(cls_preds, cls_labels):
    # Because the category prediction results are placed in the final
    # dimension, argmax must specify this dimension
    return float((cls_preds.argmax(axis=-1) == cls_labels).sum())

def bbox_eval(bbox_preds, bbox_labels, bbox_masks):
    return float((np.abs((bbox_labels - bbox_preds) * bbox_masks)).sum())

class Accumulator(object):
    """Sum a list of numbers over time"""
    def __init__(self, n):
        self.data = [0.0] * n
    def add(self, *args):
        self.data = [a+b for a, b in zip(self.data, args)]
    def reset(self):
        self.data = [0] * len(self.data)
    def __getitem__(self, i):
        return self.data[i]

def predict(X):
    anchors, cls_preds, bbox_preds = net(X.as_in_context(ctx[0]))
    cls_probs = npx.softmax(cls_preds).transpose(0, 2, 1)
    output = npx.multibox_detection(cls_probs, bbox_preds, anchors)
    idx = [i for i, row in enumerate(output[0]) if row[0] != -1]
    return output[0, idx]

if __name__ == "__main__":
    npx.set_np()
    # Anchor box的大小
    sizes = [[0.2, 0.272], [0.37, 0.447], [0.54, 0.619], [0.71, 0.79], [0.88, 0.961]]
    ratios = [[1, 2, 0.5]] * 5

    # n + m - 1，只对包含s1或者r1的感兴趣
    num_anchors = len(sizes[0]) + len(ratios[0]) - 1
    img_dir = "E:/Dataset/Captcha/img/"
    save_dir = "E:/Dataset/Captcha/rec/"
    file_prefix = "rec_256_256"
    # save_dir = "./asset/dataset/"
    # file_prefix = "train"

    batch_size = 16
    train_iter = am.load_data_test(batch_size, save_dir, file_prefix)
    ctx = am.try_all_gpus()
    net = TinySSD(num_classes=10)

    net.initialize(init=init.Xavier(), ctx=ctx)
    trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.08, 'wd': 5e-4})
    cls_loss = gluon.loss.SoftmaxCrossEntropyLoss()
    bbox_loss = gluon.loss.L1Loss()

    num_epochs = 10
    for epoch in range(num_epochs):
        train_iter.reset()  # Read data from the start.
        # accuracy_sum, mae_sum, num_examples, num_labels
        metric = Accumulator(4)
        for batch in train_iter:
            X = batch.data[0].as_in_ctx(ctx[0])
            Y = batch.label[0].as_in_ctx(ctx[0])
            with autograd.record():
                # Generate multiscale anchor boxes and predict the category and offset of each
                anchors, cls_preds, bbox_preds = net(X)
                # Label the category and offset of each anchor box
                bbox_labels, bbox_masks, cls_labels = npx.multibox_target(anchors, Y, cls_preds.transpose(0, 2, 1))
                # Calculate the loss function using the predicted and labeled category and offset values
                l = calc_loss(cls_preds, cls_labels, bbox_preds, bbox_labels, bbox_masks)

            l.backward()
            trainer.step(batch_size)
            metric.add(cls_eval(cls_preds, cls_labels), cls_labels.size,
                       bbox_eval(bbox_preds, bbox_labels, bbox_masks),
                       bbox_labels.size)
        # print("Epoch %d, cls_eval=%f, cls_labels_size=%d, bbox_eval=%f, bbox_labels_size=%d" %
        #       (epoch, metric[0], metric[1], metric[2], metric[3]))

        cls_err, bbox_mae = 1 - metric[0] / metric[1], metric[2] / metric[3]
        print("Epoch %d, cls_err=%f, bbox_mae=%f" %
              (epoch, cls_err, bbox_mae))

    print('class err %.2e, bbox mae %.2e' % (cls_err, bbox_mae))

    # img = cv.imread('./0.png')
    # feature = img.astype('float32')
    img = image.imread('./out.png')
    feature = img.astype('float32')
    X = np.expand_dims(feature.transpose(2, 0, 1), axis=0)
    output = predict(X)
    for out in output:
        imgs = cv.imread('./out.png')
        am.cv_rectangle_normalized(img=imgs, pos=out[2:], normallized=True)
        cv.imshow('show', imgs)
        o = (out[2:]*256).astype(np.int32)
        print(out[0:2], o)
        cv.waitKey(0)

    # batch = train_iter.next()
    # # !!! Need to use data[0] to get ndarray
    # # print(batch.data[0].shape, batch.label[0].shape)
    #
    #
    # imgs = (batch.data[0][0:4].transpose(0, 3, 2, 1))
    # show = am.cv_multishow(imgs, 2, 2)
    # cv.imshow('img', show)
    # cv.waitKey(0)
    # cv.destroyAllWindows()
    #
    # Y1 = forward(np.zeros((2, 8, 20, 20)), cls_predictor(5, 10))
    # Y2 = forward(np.zeros((2, 16, 10, 10)), cls_predictor(3, 10))
    # print(Y1.shape, Y2.shape)

'''



net = TinySSD(num_classes=1)
net.initialize()
X = np.zeros((32, 3, 256, 256))
anchors, cls_preds, bbox_preds = net(X)
batch_size = 8
train_iter, _ = load_data_pikachu(batch_size)


class Timer(object):
    """Record multiple running times."""
    def __init__(self):
        self.times = []
        self.start()
    def start(self):
        # Start the timer
        self.start_time = time.time()
    def stop(self):
        # Stop the timer and record the time in a list
        self.times.append(time.time() - self.start_time)
        return self.times[-1]
    def avg(self):
        # Return the average time
        return sum(self.times)/len(self.times)
    def sum(self):
        # Return the sum of time
        return sum(self.times)
    def cumsum(self):
        # Return the accumuated times
        return np.array(self.times).cumsum().tolist()











def use_svg_display():
    """Use the svg format to display a plot in Jupyter."""
    display.set_matplotlib_formats('svg')

def set_figsize(figsize=(3.5, 2.5)):
    """Set the figure size for matplotlib."""
    use_svg_display()
    plt.rcParams['figure.figsize'] = figsize

def display(img, output, threshold):
    set_figsize((5, 5))
    fig = plt.imshow(img.asnumpy())
    for row in output:
        score = float(row[1])
        if score < threshold:
            continue
        h, w = img.shape[0:2]
        bbox = [row[2:6] * np.array((w, h, w, h), ctx=row.context)]
        show_bboxes(fig.axes, bbox, '%.2f' % score, 'w')
        display(img, output, threshold=0.3)


'''