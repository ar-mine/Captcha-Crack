from captcha.image import ImageCaptcha
import os
import random
import armine as am
from mxnet.gluon import data

DATASET_ROOT = "E:/Dataset/Captcha/"
IMAGE_DIR = os.path.join(DATASET_ROOT, 'img')
LABEL_DIR = os.path.join(DATASET_ROOT, 'rec')
DEFAULT_FONTS = [os.path.join('./asset', 'simhei.ttf')]
GEN_NUM = 25000
NUM_CLASSES = 10
BATCH_SIZE = 32
DICTSET = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
           'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j',
           'k', 'l', 'm', 'n', 'p', 'q', 'r', 's', 't', 'u',
           'v', 'w', 'x', 'y', 'z', 'A', 'B', 'D', 'E', 'F',
           'G', 'H', 'L', 'M', 'N', 'Q', 'R', 'T', 'Y']


GEN_FLAG = False
READ_FLAG = True
TRAIN = False
PREDICT = False


cap = ImageCaptcha(width=200, height=100, fonts=DEFAULT_FONTS)

# 是否生成数据
if GEN_FLAG:
    f = open(os.path.join(LABEL_DIR, "label.txt"), "w")
    for i in range(GEN_NUM):
        word = ""
        for j in range(4):
            word += DICTSET[random.randint(0, NUM_CLASSES-1)]
        cap.write(word, os.path.join(IMAGE_DIR, '%d.png' % i))
        f.write(word+"\n")
    f.close()


# train_set = am.CapDataset(root=DATASET_ROOT)
# transform_test = data.vision.transforms.Compose([am.g_totensor, am.g_normalize])
# train_iter = data.DataLoader(train_set.transform_first(transform_test), BATCH_SIZE, shuffle=True, last_batch='keep')

for X, Y in train_iter:
    print(X.shape, Y.shape)



