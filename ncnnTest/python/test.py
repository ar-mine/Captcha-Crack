from mxnet import gluon, image, npx, np, ndarray
npx.set_np()

# net = gluon.SymbolBlock.imports(symbol_file='test-symbol.json',
#                                 input_names=['data'], param_file='test-0010.params', ctx=npx.gpu(0))
# m = image.imread("test2.png", 0)
# m = image.imresize(m, 28, 28)
# g_totensor = gluon.data.vision.transforms.ToTensor()
# m = g_totensor(m)
# X = np.expand_dims(m, axis=0)
# Y = np.zeros((1, 1, 28, 28))
# out = net(Y.as_in_ctx(npx.gpu(0)))
# print(out)



net1 = gluon.model_zoo.vision.resnet18_v1(prefix='resnet', pretrained=True)
net1.hybridize()
x = np.random.normal(size=(1, 3, 32, 32))
out1 = net1(x)
net1.export('net1', epoch=1)
net2 = gluon.SymbolBlock.imports('net1-symbol.json', ['data'], 'net1-0001.params')
out2 = net2(x)

