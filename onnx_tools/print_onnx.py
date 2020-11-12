import onnx
from onnx import optimizer, shape_inference, helper, numpy_helper, shape_inference
import sys

if len(sys.argv) != 2:
    print("usage: python infer_onnx.py [onnx_file]")
    exit()

model_path = sys.argv[1]
original_model = onnx.load(model_path)
#onnx.checker.check_model(original_model)

#print(original_model.graph)

def get_inputs(model):
    initializers = model.graph.initializer
    inputs = model.graph.input
    list_inputs = []
    for input in inputs:
        if not input in initializers:
            list_inputs.append(input.name)

    return list_inputs

def print_model_info(original_model):
#print("model = {}".format(original_model.graph))
    print(onnx.helper.printable_graph(original_model.graph))
    print("IR_version = {}".format(original_model.ir_version))
    print("Model_version = {}".format(original_model.model_version))
    print("len = {}".format(len(original_model.opset_import)))
    print("opset_version = {}".format(original_model.opset_import))
#    print("inputs = {}".format(get_inputs(original_model)))
    #print("op_set = {}".format(original_model.op_set))
    #print("model_version = {}".format(original_model.model_version))
    #inferred_model = shape_inference.infer_shapes(original_model)
    #print(inferred_model)

def get_attribute(node, attr_name, default_value=None):
    found = [attr for attr in node.attribute if attr.name == attr_name]
    if found:
        return helper.get_attribute_value(found[0])
    return default_value

def print_initializers(model):
    initializers = dict([(i.name, i) for i in model.graph.initializer])
    for k, v in initializers.items():
        array_v = numpy_helper.to_array(v)
        print("ini_name = {}, val = {}".format(k, array_v))

def print_constant_node(model):
    for node in model.graph.node:
        if node.op_type == "Constant":
            t = get_attribute(node, 'value')
            print("value_numpy = {}".format(numpy_helper.to_array(t)))
            print("value = {}".format(t))

            for on in node.output:
                print("output = {}".format(on))




print_model_info(original_model)
print_constant_node(original_model)
print_initializers(original_model)
