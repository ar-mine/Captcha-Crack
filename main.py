import mxnet as mx
from cnn import CNN


batch_size = 128

alexnet = CNN(net_name=1, data_name=1, batch_size=batch_size)

# alexnet.test(224)
alexnet.train(lr=0.01, num_epochs=10, ctx=alexnet.ctx)


def predict():
    # load model
    sym, arg_params, aux_params = mx.model.load_checkpoint("fashion_mnist", 1)  # load with net name and epoch num
    mod = mx.mod.Module(symbol=sym, context=mx.gpu(0), data_names=["data"], label_names=[])  # label can be empty
    mod.bind(for_training=False, data_shapes=[("data", (1, 2))])  # data shape, 1 x 2 vector for one test data record
    mod.set_params(arg_params, aux_params)

    # predict
    #predict_stress = mod.predict(eval_iter, num_batch=1)

    #return predict_stress
