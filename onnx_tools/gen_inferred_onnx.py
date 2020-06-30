import onnx
from onnx import optimizer, shape_inference
import sys

if len(sys.argv) != 3:
    print("usage: python read_onnx.py [onnx_file] [output_file]")
    exit()

model_path = sys.argv[1]
out_file_path = sys.argv[2]
original_model = onnx.load(model_path)
onnx.checker.check_model(original_model)

#print(original_model)
inferred_model = shape_inference.infer_shapes(original_model)
print('Mode graph is:\n#{}\n'.format(inferred_model))

print(onnx.helper.printable_graph(inferred_model.graph))

#onnx.save(inferred_model, out_file_path)
