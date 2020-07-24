import armine as am
from mxnet import gluon
import os
import pandas as pd

train_dir, test_dir, batch_size = 'train', 'test', 128
data_dir, label_file = 'E:/Dataset/cifar-10', 'trainLabels.csv'
input_dir, valid_ratio = 'train_valid_test', 0.1

# Only need to use once due to the large amount of time consumed
# am.reorg_cifar10_data(data_dir, label_file, train_dir, test_dir, input_dir, valid_ratio)


transform_train = gluon.data.vision.transforms.Compose([am.g_resize, am.g_re_crop, am.g_flip_rl, am.g_totensor, am.g_normalize])
transform_test = gluon.data.vision.transforms.Compose([am.g_totensor, am.g_normalize])


# Read the original image file. Flag=1 indicates that the input image has three channels (color)
train_ds = gluon.data.vision.ImageFolderDataset(os.path.join(data_dir, input_dir, 'train'), flag=1)
valid_ds = gluon.data.vision.ImageFolderDataset(os.path.join(data_dir, input_dir, 'valid'), flag=1)
train_valid_ds = gluon.data.vision.ImageFolderDataset(os.path.join(data_dir, input_dir, 'train_valid'), flag=1)
test_ds = gluon.data.vision.ImageFolderDataset(os.path.join(data_dir, input_dir, 'test'), flag=1)


train_iter = gluon.data.DataLoader(train_ds.transform_first(transform_train), batch_size, shuffle=True, last_batch='keep')
valid_iter = gluon.data.DataLoader(valid_ds.transform_first(transform_test), batch_size, shuffle=True, last_batch='keep')
train_valid_iter = gluon.data.DataLoader(train_valid_ds.transform_first(transform_train), batch_size, shuffle=True, last_batch='keep')
test_iter = gluon.data.DataLoader(test_ds.transform_first(transform_test), batch_size, shuffle=False, last_batch='keep')

loss = gluon.loss.SoftmaxCrossEntropyLoss()

ctx, num_epochs, lr, wd = am.try_gpu(), 20, 0.1, 5e-4
lr_period, lr_decay, net = 80, 0.1, am.get_net(ctx)
net.hybridize()
am.train(net, train_valid_iter, None, num_epochs, lr, wd, ctx, lr_period, lr_decay, loss, batch_size)

preds = []
net.hybridize()
for X, _ in test_iter:
    y_hat = net(X.as_in_context(ctx))
    preds.extend(y_hat.argmax(axis=1).astype(int).asnumpy())
sorted_ids = list(range(1, len(test_ds) + 1))
sorted_ids.sort(key=lambda x: str(x))
df = pd.DataFrame({'id': sorted_ids, 'label': preds})
df['label'] = df['label'].apply(lambda x: train_valid_ds.synsets[x])
df.to_csv('submission.csv', index=False)