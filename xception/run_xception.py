import numpy as np
import onnxruntime
import onnx
from onnx import numpy_helper
from onnx import AttributeProto, TensorProto, GraphProto
import time
import sys
import argparse

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

def load_model(model_file, ep_name):
    so = onnxruntime.SessionOptions()
    so.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    #Run the model on the backend
    session = onnxruntime.InferenceSession(model_file, sess_options = so)
    ep_name = ep_name + "ExecutionProvider"
    session.set_providers([ep_name])

    #Get input_name
    inputs = session.get_inputs()
    num_inputs = len(inputs)
    print("Model {} has {} inputs".format(model_file, num_inputs))

    return session

def run_inference(session, batch_size):
    #Get input_name
    inputs = session.get_inputs()
    num_inputs = len(inputs)

    #Wrap up inputs
    input_dict = {}
    for input_index in range(num_inputs):
        name = inputs[input_index].name
        print("name = {}".format(name))
        shape = inputs[input_index].shape
#        print("shape = {}".format(shape))
#        print("batch_size = {}".format(shape[0]))

        # check dynamic shape
        for index in range(len(shape)):
            if isinstance(shape[index], str):
                shape[index] = batch_size
#        print("shape = {}".format(shape))

        input_type = inputs[input_index].type
#        print(input_type)

        np_type = get_numpy_type(input_type)

        # handle dynamic shape
        is_dynamic_shape = False
        for i in range(len(shape)):
            if shape[i] == 'None':
                is_dynamic_shape = True
                shape[i] = batch_size

        if is_dynamic_shape == True:
            print('Dynamic input shape, change shape to: {}'.format(shape))

        if np_type == np.int32 or np_type == np.int64:
#            print("integer type")
            input_dict[name] = np.ones(shape).astype(np_type)
        else:
#            print("type = {}".format(np_type))
            input_dict[name] = np.random.random(shape).astype(np_type)

#    for keys, values in input_dict.items():
#        print(keys)
#        print(values)

    # warm up run to execlude one-time build time
    outputs = session.run([], input_dict)
    start = time.time()
    outputs = session.run([], input_dict)
    end = time.time()
    exec_time = (end - start) * 1000.0;
    print("Batch_size:\t{}\texec_time =\t{}\tms".format(batch_size, exec_time))
    num_outputs = len(outputs)
#    for out_index in range(num_outputs):
#        print("output[{}]'s shape = {}".format(out_index, outputs[out_index].shape))
#        print("output[{}] = ".format(out_index))
#        print(outputs[out_index])

def main():
    parser = argparse.ArgumentParser(description="Run the xception model")
    parser.add_argument('--batch_size', type=int, metavar='batch_size', default=1, help='Specify the batch size used in the model')
    parser.add_argument('model', type=str, metavar='model_file', help='onnx file name of the model')
    parser.add_argument('--ep', type=str, metavar='ep_name', default="MIGraphX", help='Name of the execution provider, CPU or MIGraphX')
    args = parser.parse_args()

    batch_size = args.batch_size
    model_file = args.model
    ep_name = args.ep

    session = load_model(model_file, ep_name)
    run_inference(session, batch_size)

if __name__ == "__main__":
    main()

