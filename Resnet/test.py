from captcha.image import ImageCaptcha
import os
import random
import armine as am
from mxnet.gluon import data
from mxnet import gluon, init, np, autograd, image
from tqdm import tqdm
import net as anet
import config
import time

GEN_FLAG = False
TRAIN_FLAG = True
PREDICT_FLAG = True


# 是否生成数据
if GEN_FLAG:
    cap = ImageCaptcha(width=config.WIDTH, height=config.HEIGHT, fonts=config.DEFAULT_FONTS)
    f = open(os.path.join(config.LABEL_DIR, config.LABEL_NAME), "w")
    for i in tqdm(range(config.GEN_NUM)):
        word = ""
        for j in range(config.C_LEN):
            word += config.DICTSET[random.randint(0, config.NUM_CLASSES-1)]
        cap.write(word, os.path.join(config.IMAGE_DIR, '%d.png' % i))
        f.write(word+"\n")
    f.close()

# 是否进行训练
if TRAIN_FLAG:
    train_set = am.CapDataset(root=config.DATASET_ROOT, flag=1)
    transform_test = data.vision.transforms.Compose([am.g_totensor])
    train_iter = data.DataLoader(train_set.transform_first(transform_test),
                                 config.BATCH_SIZE, shuffle=False, last_batch='keep')
    net = anet.CNN(config.NUM_CLASSES, config.C_LEN).get_net()
    loss = gluon.loss.SoftmaxCrossEntropyLoss()
    ctx, num_epochs, lr, wd = am.try_gpu(), 30, 1e-4, 5e-4
    net.initialize(ctx=ctx, init=init.Xavier())
    # trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': lr, 'momentum': 0.9, 'wd': wd})
    trainer = gluon.Trainer(net.collect_params(), 'adam', {'learning_rate': lr})
    print("train begin")
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n, timer = 0.0, 0.0, 0, time.time()
        for X, y in train_iter:
            X = X.as_in_ctx(ctx)
            y = y.as_in_ctx(ctx)
            l = []
            with autograd.record():
                y_hat = net(X)
                for i, y_ in enumerate(y_hat):
                    l.append(loss(y_, y[:, i]).sum())
            autograd.backward(l)
            trainer.step(config.BATCH_SIZE)
            acc = 0
            for i, y_ in enumerate(y_hat):
                acc += float((y_.argmax(axis=1) == y[:, i]).sum())
            for l_ in l:
                train_l_sum += l_
            train_acc_sum += acc
            n += y.size
        epoch_s = ("epoch %d, loss %f, train acc %f, time %.1fs" %
                   (epoch + 1, train_l_sum / n, train_acc_sum / n, time.time()-timer))
        print(epoch_s)

# 是否进行预测
if PREDICT_FLAG:
    src = image.imread("F:/Dataset/Captcha/img/1.png", 1)
    src = image.imresize(src, w=128, h=64)
    X = am.g_totensor(src)
    X = np.expand_dims(X, axis=0)
    Y = net(X.as_in_ctx(ctx))
    print(Y)
    print(Y[0].argmax())
    print(Y[1].argmax())
    print(Y[2].argmax())
    print(Y[3].argmax())




