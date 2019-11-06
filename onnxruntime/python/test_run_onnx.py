import numpy as np
import onnxruntime
import onnx
from onnx import numpy_helper
from onnx import AttributeProto, TensorProto, GraphProto
import time
import sys

#type_table = {
#    TensorProto.INT64 : np.int64,
#    TensorProto.INT32 : np.int32,
#    TensorProto.FLOAT : np.float32
#}

type_table = {
    'tensor(int64)' : np.int64,
    'tensor(int32)' : np.int32,
    'tensor(float)' : np.float32
}

def get_numpy_type(tensor_type):
    return_type = np.float32
    if tensor_type in type_table.keys():
        return_type = type_table[tensor_type]

    return return_type

if len(sys.argv) != 2:
    print("Usage: python test_run_onnx.py file.onnx")
    exit()

model_file = sys.argv[1]

so = onnxruntime.SessionOptions()
so.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
#Run the model on the backend
session = onnxruntime.InferenceSession(model_file, sess_options = so)
session.set_providers(['MiGraphXExecutionProvider'])

#Get input_name
inputs = session.get_inputs()
num_inputs = len(inputs)
print("Model {} has {} inputs".format(model_file, num_inputs))

#Wrap up inputs
input_dict = {}
for input_index in range(num_inputs):
    name = inputs[input_index].name
    print("name = {}".format(name))
    shape = inputs[input_index].shape
    print("shape = {}".format(shape))
    input_type = inputs[input_index].type
    print(input_type)

    np_type = get_numpy_type(input_type)

    if np_type == np.int32 or np_type == np.int64:
#    if isinstance(np_type, np.int32) or isinstance(np_type, np.int64):
        print("integer type")
        input_dict[name] = np.ones(shape).astype(np_type)
    else:
        print("type = {}".format(np_type))
        input_dict[name] = np.random.random(shape).astype(np_type)

for keys, values in input_dict.items():
    print(keys)
    print(values)

outputs = session.run([], input_dict)
num_outputs = len(outputs)
for out_index in range(num_outputs):
    print("output[{}]'s shape = {}".format(out_index, outputs[out_index].shape))
    print("output[{}] = ".format(out_index))
    print(outputs[out_index])
