import cv2 as cv
import numpy as np
import os
from mxnet import autograd, contrib, gluon, image, init, npx
from IPython import display
import matplotlib.pyplot as plt

npx.set_np()
'''
    This is a library with some functions useful to this project, some are from other open-source and some are 
    created by me. So maybe you can see some familiar code in this library.
'''


def cv_rectangle_normalized(img, pos, normallized=False, text=None, color=(255, 0, 0), thickness=2):
    '''  Simplify the opencv function to draw rectangle  '''
    w = img.shape[0]
    h = img.shape[1]
    pos = np.array(pos)
    scale = np.array((h, w, h, w))
    if len(pos.shape) < 2:
        if text is None:
            text = ""
        if normallized:
            pos = (pos * scale).astype(np.int32)
        cv.rectangle(img=img, pt1=(pos[0], pos[1]), pt2=(pos[2], pos[3]),
                     color=color, thickness=thickness)
        cv_showtxt(img, text, (pos[0], pos[1]), color)
    else:
        for i, posi in enumerate(pos):
            if normallized:
                posi = (posi * scale).astype(np.int32)
            posi = posi.tolist()
            cv.rectangle(img=img, pt1=(posi[0], posi[1]), pt2=(posi[2], posi[3]),
                         color=color, thickness=thickness)
            if text is None:
                cv_showtxt(img, "", (posi[0], posi[1]), color)
            else:
                cv_showtxt(img, text[i], (posi[0], posi[1]), color)


def cv_showtxt(img, text, org, color, fontFace=cv.FONT_HERSHEY_SIMPLEX, fontScale=1, thickness=2):
    cv.putText(img=img, text=text, org=org, fontFace=fontFace,
               fontScale=fontScale, color=color, thickness=thickness)


def cv_multishow(imglist, row, column):
    max_w, max_h = 0, 0
    if len(imglist) > row*column:
        print("Error: cv_multishow -- too many imgs in imglist.")
        return -1
    for img in imglist:
        if img.shape[1] > max_w:
            max_w = img.shape[1]
        if img.shape[0] > max_h:
            max_h = img.shape[0]
    mulimg = np.zeros((max_h*row, max_w*column, 3)).astype(np.uint8)
    for i, img in enumerate(imglist):
        mulimg[int(i/column)*max_h:int(i/column)*max_h+img.shape[0], int(i%column)*max_w:(i%column)*max_w+img.shape[1], :] = img
    return mulimg


def cv_multibox():
    pass


def try_all_gpus():
    """Return all available GPUs, or [cpu(),] if no GPU exists.
       Copy from d2l library"""
    ctxes = [npx.gpu(i) for i in range(npx.num_gpus())]
    return ctxes if ctxes else [npx.cpu()]


def load_data_test(batch_size, data_dir, fname):
    train_iter = image.ImageDetIter(
        path_imgrec=os.path.join(data_dir, '%s.rec' % fname),
        path_imgidx=os.path.join(data_dir, '%s.idx' % fname),
        batch_size=batch_size,
        shuffle=True,   # Read the dataset in random order
        data_shape=(3, 256, 256)
    )
    return train_iter


# class Animator(object):
#     def __init__(self, xlabel=None, ylabel=None, legend=[], xlim=None,
#                  ylim=None, xscale='linear', yscale='linear', fmts=None,
#                  nrows=1, ncols=1, figsize=(3.5, 2.5)):
#         """Incrementally plot multiple lines."""
#         d2l.use_svg_display()
#         self.fig, self.axes = d2l.plt.subplots(nrows, ncols, figsize=figsize)
#         if nrows * ncols == 1:
#         self.axes = [self.axes, ]
#         # Use a lambda to capture arguments
#         self.config_axes = lambda: d2l.set_axes(
#         self.axes[0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
#         self.X, self.Y, self.fmts = None, None, fmts
#     def add(self, x, y):
#         """Add multiple data points into the figure."""
#         if not hasattr(y, "__len__"):
#             y = [y]
#         n = len(y)
#         if not hasattr(x, "__len__"):
#         x = [x] * n
#         if not self.X:
#         self.X = [[] for _ in range(n)]
#         if not self.Y:
#         self.Y = [[] for _ in range(n)]
#         if not self.fmts:
#         self.fmts = ['-'] * n
#         for i, (a, b) in enumerate(zip(x, y)):
#         if a is not None and b is not None:
#         self.X[i].append(a)
#         self.Y[i].append(b)
#         self.axes[0].cla()
#         for x, y, fmt in zip(self.X, self.Y, self.fmts):
#         self.axes[0].plot(x, y, fmt)
#         self.config_axes()
#         display.display(self.fig)
#         display.clear_output(wait=True)

if __name__ == "__main__":
    img = image.imread("./asset/catdog.jpg")
    img = img.asnumpy()
    cv.imshow("test", img)
    cv.waitKey(0)
