import onnx
from onnx import optimizer, shape_inference
from onnx.tools import update_model_dims
import sys

if len(sys.argv) != 3:
    print("usage: python infer_onnx.py [onnx_file] [out_file]")
    exit()

model_path = sys.argv[1]
original_model = onnx.load(model_path)
print("Original graph = ")
print(onnx.helper.printable_graph(original_model.graph))
print("IR_version = {}".format(original_model.ir_version))

# Here both 'seq', 'batch' and -1 are dynamic using dim_param.
#variable_length_model = update_model_dims.update_inputs_outputs_dims(original_model, {'0': [1, 'batch', 32, 32], '1' : [1, 'batch', 5, 5], '2' : [1]}, {'5', [1, 'batch', 28, 28]})
variable_length_model = update_model_dims.update_inputs_outputs_dims(original_model, {'0': [1, 'batch', 32, 32], '1' : [1, 'batch', 5, 5], '2' : [1]}, {'5': [1, 'batch', 14, 14]})
print("variable graph = ")
print(onnx.helper.printable_graph(variable_length_model.graph))
onnx.save(variable_length_model, sys.argv[2])

#print("op_set = {}".format(original_model.op_set))
#print("model_version = {}".format(original_model.model_version))
#inferred_model = shape_inference.infer_shapes(original_model)
#print(inferred_model)
