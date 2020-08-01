from mxnet import autograd, np, npx, gluon, init
from onnx import checker
import onnx
from mxnet.contrib import onnx as onnx_mxnet
npx.set_np()


sym = './test-symbol.json'
params = './test-0010.params'
input_shape = (1, 1, 28, 28)
onnx_file = './test.onnx'
converted_model_path = onnx_mxnet.export_model(sym, params, [input_shape], np.float32, onnx_file, 1)

# Load onnx model
model_proto = onnx.load_model(converted_model_path)

# Check if converted ONNX protobuf is valid
checker.check_graph(model_proto.graph)