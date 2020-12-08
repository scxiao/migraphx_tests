from io import open
import string
import re
import random
import sys
import migraphx
import time
import numpy as np

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

def check_lines(model, scratch_output, target_name, batch_lines):
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
 
    model_param_map = {}
    args = []
    for key, value in model.get_parameter_shapes().items():
        if key in scratch_output:
            model_param_map[key] = scratch_output[key]
        else:
            if key in input_map:
                args.append(np.array(input_map[key], dtype = np.longlong).reshape(value.lens()))
            else:
                hash_val = hash(key) % (2 ** 32 - 1)
                np.random.seed(hash_val)
                args.append(np.random.randn(value.elements()).astype(np.single).reshape(value.lens()))
            model_param_map[key] = migraphx.argument(args[-1])

    result = np.array(model.run(model_param_map))

    print("{}".format(result))
    output = np.squeeze(np.reshape(result, (1, -1)), axis=0).tolist()

    correct_num = 0;
    for i in range(len(batch_lines)):
        calc_label = 0 if output[2 * i] >= output[2 * i + 1] else 1
        res = 1 if calc_label == batch_labels[i] else 0
        correct_num += res

    return correct_num

def main():
    if len(sys.argv) != 4:
        print("Usage: python mrpc.py file.onnx input.tsv cpu/gpu")
        exit(1)

    # load model
    model = migraphx.parse_onnx(sys.argv[1])

    # read input file
    lines = open(sys.argv[2]).read().strip().split('\n')

    # get target name
    target_name = "cpu"
    if sys.argv[3] == "gpu":
        target_name = "gpu"

    model.compile(migraphx.get_target(target_name))
    batch_size = model.get_parameter_shapes()["input.1"].lens()[0]

    # wrap up output and scratch memory to be reused
    scratch_output = {}
    for key, value in model.get_parameter_shapes().items():
        if key == "scratch" or key == "output":
            hash_val = hash(key) % (2 ** 32 - 1)
            np.random.seed(hash_val)
            scratch_output[key] = migraphx.argument(np.random.randn(value.elements()).astype(np.single).reshape(value.lens()))

    # first line is useless information
    accu_num = 0;
    indices = list(range(len(lines) - 1))
    indices = indices[1:(len(lines) - 1) : batch_size]
    start = time.time()
    for i in indices:
        chk_res = check_lines(model, scratch_output, target_name, lines[i : i + batch_size])
        accu_num += chk_res
    end = time.time()
    
    accu_rate = 1.0 * accu_num / (len(lines) - 1)
    print("accuracy rate = {}".format(accu_rate))
    print("elapsed time = {}".format(end - start))

if __name__ == "__main__":
    main()
