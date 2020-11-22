import numpy as np
import onnxruntime
import onnx
from onnx import numpy_helper
from onnx import AttributeProto, TensorProto, GraphProto
import time
import sys
import os
import argparse

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

def write_tensor_to_file(data, out_dir, index, is_input):
    # convert numpy array to onnx tensor
    tensor = numpy_helper.from_array(data)
    data_str = tensor.SerializeToString()
    name_prefix = out_dir + '/'
    if not os.path.isdir(name_prefix):
        os.mkdir(name_prefix)
    if is_input:
        name_prefix = name_prefix + 'input_'
    else:
        name_prefix = name_prefix + 'output_'

    filename = name_prefix + str(index) + '.pb'
    file = open(filename, 'wb')
    file.write(data_str)
    file.close()

def copy_model(model_file, dst_dir):
    if not os.path.isdir(dst_dir):
        os.mkdir(dst_dir)
    file_name = os.path.basename(model_file)
    dst_file = dst_dir + '/' + file_name
    cmd = 'cp ' + model_file + ' ' + dst_file
    os.system(cmd)

def run_inference(session, default_batch_size, test_dir):
    #Get input_name
    inputs = session.get_inputs()
    num_inputs = len(inputs)

    data_dir = test_dir + '/test_data_set_0'

    #Wrap up inputs
    input_dict = {}
    for input_index in range(num_inputs):
        name = inputs[input_index].name
        print("Input parameter: {}".format(name))
        shape = inputs[input_index].shape
        print("shape = {}".format(shape))
        print("batch_size = {}".format(shape[0]))

        # check dynamic shape
        for index in range(len(shape)):
            if isinstance(shape[index], str):
                shape[index] = default_batch_size
        print("shape = {}".format(shape))

        input_type = inputs[input_index].type
        print(input_type)

        np_type = get_numpy_type(input_type)

        # handle dynamic shape
        is_dynamic_shape = False
        for i in range(len(shape)):
            if shape[i] == 'None':
                is_dynamic_shape = True
                shape[i] = default_batch_size

        if is_dynamic_shape == True:
            print('Dynamic input shape, change shape to: {}'.format(shape))

        if np_type == np.int32 or np_type == np.int64:
            print("integer type")
            input_dict[name] = np.ones(shape).astype(np_type)
        else:
            print("type = {}".format(np_type))
            input_dict[name] = np.random.random(shape).astype(np_type)

    index = 0
    for keys, values in input_dict.items():
        print(keys)
        print(values)
        write_tensor_to_file(values, data_dir, index, True)
        index = index + 1

    outputs = session.run([], input_dict)
    num_outputs = len(outputs)
    for out_index in range(num_outputs):
        print("output[{}]'s shape = {}".format(out_index, outputs[out_index].shape))
        print("output[{}] = ".format(out_index))
        print(outputs[out_index])
        write_tensor_to_file(outputs[out_index], data_dir, out_index, False)

def parse_args():
    parser = argparse.ArgumentParser(description="Run the xception model")
    parser.add_argument('--batch_size', type=int, metavar='batch_size', default=1, help='Specify the batch size used in the model')
    parser.add_argument('model', type=str, metavar='model_file', help='onnx file name of the model')
    parser.add_argument('--ep', type=str, metavar='ep_name', default="MIGraphX", help='Name of the execution provider, CPU or MIGraphX')
    parser.add_argument('--create_test', type=bool, metavar='create_test', default=bool, help='Creat a unit test for the run')
    parser.add_argument('--case_dir', type=str, default='ort_test', help='folder where the created test is stored')
    parser.add_argument('--case_num', type=int, metavar='case_num', default=1, help='Number of cases')
    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    batch_size = args.batch_size
    model_file = args.model
    ep_name = args.ep
    test_dir = args.case_dir
    if not args.case_dir:
        test_dir = 'example'

    # copy model from source to distination
    copy_model(model_file, test_dir)

    session = load_model(model_file, ep_name)
    run_inference(session, batch_size, test_dir)


if __name__ == "__main__":
    main()

