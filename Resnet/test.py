from captcha.image import ImageCaptcha
import os
import random
import armine as am
from mxnet.gluon import data
from mxnet import gluon, init, np, autograd, image
from tqdm import tqdm
import net


DATASET_ROOT = "E:/Dataset/Captcha/"
IMAGE_DIR = os.path.join(DATASET_ROOT, 'img')
LABEL_DIR = os.path.join(DATASET_ROOT, 'rec')
DEFAULT_FONTS = [os.path.join('./asset/font', 'simhei.ttf')]
GEN_NUM = 25000
NUM_CLASSES = 10
BATCH_SIZE = 16
DICTSET = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
           'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j',
           'k', 'l', 'm', 'n', 'p', 'q', 'r', 's', 't', 'u',
           'v', 'w', 'x', 'y', 'z', 'A', 'B', 'D', 'E', 'F',
           'G', 'H', 'L', 'M', 'N', 'Q', 'R', 'T', 'Y']


GEN_FLAG = False
READ_FLAG = False
TRAIN = False
PREDICT = False


cap = ImageCaptcha(width=200, height=100, fonts=DEFAULT_FONTS)

# 是否生成数据
if GEN_FLAG:
    f = open(os.path.join(LABEL_DIR, "label.txt"), "w")
    for i in tqdm(range(GEN_NUM)):
        word = ""
        for j in range(4):
            word += DICTSET[random.randint(0, NUM_CLASSES-1)]
        cap.write(word, os.path.join(IMAGE_DIR, '%d.png' % i))
        f.write(word+"\n")
    f.close()


train_set = am.CapDataset(root=DATASET_ROOT)
transform_test = data.vision.transforms.Compose([am.g_totensor, am.g_normalize])
train_iter = data.DataLoader(train_set.transform_first(transform_test), BATCH_SIZE, shuffle=True, last_batch='keep')

cnn = net.CNN(NUM_CLASSES)
loss = gluon.loss.SoftmaxCrossEntropyLoss()
ctx, num_epochs, lr, wd = am.try_gpu(), 15, 0.005, 5e-4
lr_period, lr_decay, net = 80, 0.1, cnn.get_net()
net.initialize(ctx=ctx, init=init.Xavier())
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': lr, 'momentum': 0.9, 'wd': wd})

print("train begin")
for epoch in range(num_epochs):
    train_l_sum, train_acc_sum, n = 0.0, 0.0, 0
    for X, y in train_iter:
        X = X.as_in_context(ctx)
        y = am.encoder(y, NUM_CLASSES, c_len=4, ctx=ctx).as_in_context(ctx)
        with autograd.record():
            y_hat = net(X)
            l = am.decode_loss(y, y_hat, NUM_CLASSES, 4, loss)
        l.backward()
        trainer.step(BATCH_SIZE)
        train_l_sum += float(l)
#        train_acc_sum += float((y_hat.argmax(axis=1) == y).sum())
#        n += y.size
    # epoch_s = ("epoch %d, loss %f, train acc %f, " %
    #            (epoch + 1, train_l_sum / n, train_acc_sum / n))
    # print(epoch_s + ', lr ' + str(trainer.learning_rate))
    print(epoch)


src = image.imread("E:/Dataset/Captcha/img/4.png").as_in_context(ctx)
X = am.g_totensor(src)
X = am.g_normalize(X)
X = np.expand_dims(X, axis=0)
Y = net(X)
print(Y[0, 0:10].argmax())
print(Y[0, 10:20].argmax())
print(Y[0, 20:30].argmax())
print(Y[0, 30:40].argmax())




