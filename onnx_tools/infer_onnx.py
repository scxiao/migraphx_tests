import onnx
from onnx import optimizer, shape_inference
import sys

if len(sys.argv) != 2:
    print("usage: python infer_onnx.py [onnx_file]")
    exit()

model_path = sys.argv[1]
original_model = onnx.load(model_path)
#onnx.checker.check_model(original_model)

#print(original_model.graph)

print(onnx.helper.printable_graph(original_model.graph))
print("IR_version = {}".format(original_model.ir_version))
print("Model_version = {}".format(original_model.model_version))
print("len = {}".format(len(original_model.opset_import)))
print("opset_version = {}".format(original_model.opset_import))
#print("op_set = {}".format(original_model.op_set))
#print("model_version = {}".format(original_model.model_version))
#inferred_model = shape_inference.infer_shapes(original_model)
#print(inferred_model)
