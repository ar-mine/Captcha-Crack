from mxnet import gluon, init, nd, autograd
from mxnet.gluon import loss as gloss, utils as gutils, nn
import mxnet as mx
from dataset import Dataset
import time

class CNN:
    def __init__(self, net_name, data_name, batch_size):
        if net_name == 0:
            pass
        elif net_name == 1:
            self.net = alexnet_init()
        self.ctx = try_gpu()
        self.batch_size = batch_size
        fm_data = Dataset(data_name, self.batch_size, resize=224)
        self.train_iter, self.test_iter = fm_data.train_iter, fm_data.test_iter



    def test(self, *args):
        '''根据输入数据形状输出各层数参数,默认输入宽度与高度相同'''
        self.net.initialize()
        w = args[0]
        if len(args) == 1:
            h = w
        elif len(args) == 2:
            h = args[1]
        else:
            print("参数过多")
            exit(-1)
        X = nd.random.uniform(shape=(1, 1, w, h))
        for layer in self.net:
            in_array = layer(X)
            print(layer.name, 'output shape:\t', X.shape)

    def train(self, lr, num_epochs, ctx):
        self.net.initialize(force_reinit=True, ctx=ctx, init=init.Xavier())
        trainer = gluon.Trainer(self.net.collect_params(), 'sgd', {'learning_rate': lr})
        train_ch5(self.net, self.train_iter, self.test_iter, self.batch_size, trainer, ctx, num_epochs)


def try_gpu():
    """If GPU is available, return mx.gpu(0); else return mx.cpu()."""
    try:
        ctx = mx.gpu()
        _ = nd.array([0], ctx=ctx)
    except mx.base.MXNetError:
        ctx = mx.cpu()
    return ctx


def train_ch5(net, train_iter, test_iter, batch_size, trainer, ctx,
              num_epochs):
    """Train and evaluate a model with CPU or GPU."""
    print('training on', ctx)
    loss = gloss.SoftmaxCrossEntropyLoss()
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n, start = 0.0, 0.0, 0, time.time()
        for X, y in train_iter:
            X, y = X.as_in_context(ctx), y.as_in_context(ctx)
            with autograd.record():
                y_hat = net(X)
                l = loss(y_hat, y).sum()
            l.backward()
            trainer.step(batch_size)
            y = y.astype('float32')
            train_l_sum += l.asscalar()
            train_acc_sum += (y_hat.argmax(axis=1) == y).sum().asscalar()
            n += y.size
        test_acc = evaluate_accuracy(test_iter, net, ctx)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f, '
              'time %.1f sec'
              % (epoch + 1, train_l_sum / n, train_acc_sum / n, test_acc,
                 time.time() - start))

    T = nd.random.uniform(shape=(1, 1, 224, 224), ctx=mx.gpu(0))
    net.hybridize()
    net(T)
    net.export(path=".\\asset\\model\\fashion_mnist", epoch=num_epochs)


def evaluate_accuracy(data_iter, net, ctx=[mx.cpu()]):
    """Evaluate accuracy of a model on the given data set."""
    if isinstance(ctx, mx.Context):
        ctx = [ctx]
    acc_sum, n = nd.array([0]), 0
    for batch in data_iter:
        features, labels, _ = _get_batch(batch, ctx)
        for X, y in zip(features, labels):
            y = y.astype('float32')
            acc_sum += (net(X).argmax(axis=1) == y).sum().copyto(mx.cpu())
            n += y.size
        acc_sum.wait_to_read()
    return acc_sum.asscalar() / n


def _get_batch(batch, ctx):
    """Return features and labels on ctx."""
    features, labels = batch
    if labels.dtype != features.dtype:
        labels = labels.astype(features.dtype)
    return (gutils.split_and_load(features, ctx),
            gutils.split_and_load(labels, ctx), features.shape[0])


def alexnet_init():
    net = nn.HybridSequential()
    # 使用较大的11 x 11窗口来捕获物体。同时使用步幅4来较大幅度减小输出高和宽。这里使用的输出通
    # 道数比LeNet中的也要大很多
    net.add(nn.Conv2D(96, kernel_size=11, strides=4, activation='relu'),
            nn.MaxPool2D(pool_size=3, strides=2),
            # 减小卷积窗口，使用填充为2来使得输入与输出的高和宽一致，且增大输出通道数
            nn.Conv2D(256, kernel_size=5, padding=2, activation='relu'),
            nn.MaxPool2D(pool_size=3, strides=2),
            # 连续3个卷积层，且使用更小的卷积窗口。除了最后的卷积层外，进一步增大了输出通道数。
            # 前两个卷积层后不使用池化层来减小输入的高和宽
            nn.Conv2D(384, kernel_size=3, padding=1, activation='relu'),
            nn.Conv2D(384, kernel_size=3, padding=1, activation='relu'),
            nn.Conv2D(256, kernel_size=3, padding=1, activation='relu'),
            nn.MaxPool2D(pool_size=3, strides=2),
            # 这里全连接层的输出个数比LeNet中的大数倍。使用丢弃层来缓解过拟合
            nn.Dense(4096, activation="relu"), nn.Dropout(0.5),
            nn.Dense(4096, activation="relu"), nn.Dropout(0.5),
            # 输出层。由于这里使用Fashion-MNIST，所以用类别数为10，而非论文中的1000
            nn.Dense(10))
    return net

