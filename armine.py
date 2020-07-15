import cv2 as cv
import numpy as np
from mxnet import npx

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


def cv_rectangle_normalized(img, pos, text=None, color=(255, 0, 0), thickness=2):
    '''  Simplify the opencv function to draw rectangle  '''
    pos = np.array(pos)
    w = img.shape[0]
    h = img.shape[1]
    scale = np.array((h, w, h, w))
    if type(pos[0]) is not np.ndarray:
        if text is None:
            text = ""
        pos = (pos * scale).astype(np.int32)
        pos.astype(np.int32)
        cv.rectangle(img=img, pt1=(pos[0], pos[1]), pt2=(pos[2], pos[3]),
                     color=color, thickness=thickness)
        cv_showtxt(img, text, (pos[0], pos[1]), color)
    else:
        for i, posi in enumerate(pos):
            posi = (posi * scale).astype(np.int32)
            cv.rectangle(img=img, pt1=(posi[0], posi[1]), pt2=(posi[2], posi[3]),
                         color=color, thickness=thickness)
            if text is None:
                cv_showtxt(img, "", (posi[0], posi[1]), color)
            else:
                cv_showtxt(img, text[i], (posi[0], posi[1]), color)


def cv_rectangle(img, pos, text="", color=(255, 0, 0), thickness=2):
    '''  Simplify the opencv function to draw rectangle  '''
    pos = np.array(pos)
    if type(pos[0]) is not list:
        cv.rectangle(img=img, pt1=(pos[0], pos[1]), pt2=(pos[2], pos[3]),
                     color=color, thickness=thickness)
        cv_showtxt(img, text, (pos[0], pos[1]), color)
    else:
        for i, posi in enumerate(pos):
            cv.rectangle(img=img, pt1=(posi[0], posi[1]), pt2=(posi[2], posi[3]),
                         color=color, thickness=thickness)
            cv_showtxt(img, text[i], (posi[0], posi[1]), color)


def cv_showtxt(img, text, org, color, fontFace=cv.FONT_HERSHEY_SIMPLEX, fontScale=1, thickness=2):
    cv.putText(img=img, text=text, org=org, fontFace=fontFace,
               fontScale=fontScale, color=color, thickness=thickness)


# cv_rectangle_normalized(img, ground_truth[:, 1:], ["dog", "cat"])
#
# cv.imshow('img', img)
# cv.waitKey(0)
# cv.destroyAllWindows()

labels = npx.multibox_target(np.expand_dims(anchors, axis=0),
                            np.expand_dims(ground_truth, axis=0),
                            np.zeros((1, 3, 5)))

