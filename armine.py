import cv2 as cv
import numpy as np
import os
from mxnet import autograd, contrib, gluon, image, init, npx

npx.set_np()
'''
    This is a library with some functions useful to this project, some are from other open-source and some are 
    created by me. So maybe you can see some familiar code in this library.
'''
img = cv.imread("./asset/catdog.jpg")
dog_bbox, cat_bbox = [60, 45, 378, 516], [400, 112, 655, 493]
ground_truth = np.array([[0, 0.1, 0.08, 0.52, 0.92], [1, 0.55, 0.2, 0.9, 0.88]])
anchors = np.array([[0, 0.1, 0.2, 0.3], [0.15, 0.2, 0.4, 0.4],
                   [0.63, 0.05, 0.88, 0.98], [0.66, 0.45, 0.8, 0.8], [0.57, 0.3, 0.92, 0.9]])


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
            pos = (pos * scale).astype(np.uint8)
        pos.astype(np.int32)
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
# cv_rectangle_normalized(img, ground_truth[:, 1:], ["dog", "cat"])
#
# cv.imshow('img', img)
# cv.waitKey(0)
# cv.destroyAllWindows()

# labels = npx.multibox_target(np.expand_dims(anchors, axis=0),
#                             np.expand_dims(ground_truth, axis=0),
#                             np.zeros((1, 3, 5)))

