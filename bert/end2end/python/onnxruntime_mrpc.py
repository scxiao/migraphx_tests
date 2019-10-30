from io import open
import string
import re
import random
import sys
import time
import numpy as np
import onnxruntime
from onnx import numpy_helper

def parse_line(line):
    tokens = line.strip().split('\t')
    label = int(tokens[0])
    token_ids = list(map(int, tokens[1].strip().split(',')))
    token_ids.insert(0, 101)
    token_ids.append(102)
    sent_id = [0] * len(token_ids)
    sent2_token_ids = list(map(int, tokens[2].strip().split(',')))
    sent2_token_ids.append(102)
    sent2_id = [1] * len(sent2_token_ids)
    token_ids.extend(sent2_token_ids)
    sent_id.extend(sent2_id)
    seg_id = [1] * len(sent_id)

    if len(sent_id) < 128:
        patch = [0] * (128 - len(sent_id))
        token_ids.extend(patch)
        sent_id.extend(patch)
        seg_id.extend(patch)

    return (label, token_ids, sent_id, seg_id)

def check_lines(sess, batch_lines, input_shape):

    input_map = {"input.1": [],
                 "input.3": [],
                 "2": []}
    batch_labels = []
    for l in batch_lines:
        label, token_ids, sent_id, seg_id = parse_line(l)
        input_map["input.1"].extend(token_ids)
        input_map["input.3"].extend(sent_id)
        input_map["2"].extend(seg_id)
        batch_labels.append(label)

    input_map["input.1"] = np.reshape(np.array(input_map["input.1"], dtype = np.longlong), input_shape["input.1"])
    input_map["input.3"] = np.reshape(np.array(input_map["input.3"], dtype = np.longlong), input_shape["input.3"])
    input_map["2"] = np.reshape(np.array(input_map["2"], dtype = np.longlong), input_shape["2"])

    outputs = sess.run([], input_map)
    result = outputs[0]

    print("{}".format(result))
    output = np.squeeze(np.reshape(result, (1, -1)), axis=0).tolist()

    correct_num = 0;
    for i in range(len(batch_lines)):
        calc_label = 0 if output[2 * i] >= output[2 * i + 1] else 1
        res = 1 if calc_label == batch_labels[i] else 0
        correct_num += res

    return correct_num

def main():
    if len(sys.argv) != 3:
        print("Usage: python onnxruntime_mrpc.py file.onnx input.tsv")
        exit(1)

    # load model
    input_shape = {}
    session = onnxruntime.InferenceSession(sys.argv[1])
    inputs = session.get_inputs()
    num_inputs = len(inputs)
    for in_index in range(num_inputs):
        name = inputs[in_index].name
        print("name = {}".format(name))
        shape = inputs[in_index].shape
        print("shape = {}".format(shape))
        input_type = inputs[in_index].type
        print(input_type)
        input_shape[name] = shape

    # read input file
    lines = open(sys.argv[2]).read().strip().split('\n')

    batch_size = session.get_inputs()[0].shape[0]

    # first line is useless information
    accu_num = 0;
    indices = list(range(len(lines) - 1))
    indices = indices[1:(len(lines) - 1) : batch_size]
    start = time.time()
    for i in indices:
        chk_res = check_lines(session, lines[i : i + batch_size], input_shape)
        accu_num += chk_res
    end = time.time()
    
    accu_rate = 1.0 * accu_num / (len(lines) - 1)
    print("accuracy rate = {}".format(accu_rate))
    print("elapsed time = {}".format(end - start))

if __name__ == "__main__":
    main()
