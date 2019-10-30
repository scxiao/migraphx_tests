from io import open
import string
import re
import random
import sys
import migraphx
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

def check_line(model, target_name, l):
    label, token_ids, sent_id, seg_id = parse_line(l)
    input_map = {"input.1": token_ids,
                 "input.3": sent_id,
                 "2": seg_id}

    model_param_map = {}
    args = []
    for key, value in model.get_parameter_shapes().items():
        if key in input_map:
            args.append(np.array(input_map[key], dtype = np.longlong).reshape(value.lens()))
        else:
            hash_val = hash(key) % (2 ** 32 - 1)
            np.random.seed(hash_val)
            args.append(np.random.randn(value.elements()).astype(np.single).reshape(value.lens()))

        if target_name == "gpu":
            model_param_map[key] = migraphx.to_gpu(migraphx.argument(args[-1]))
        else:
            model_param_map[key] = migraphx.argument(args[-1])

    if target_name == "gpu":
        result = np.array(migraphx.from_gpu(model.run(model_param_map)))
    else:
        result = np.array(model.run(model_param_map))

    output = np.squeeze(result, axis=0).tolist()
    print("output = {}".format(output))

    calc_label = 0 if output[0] >= output[1] else 1

    return 1 if calc_label == label else 0

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

    # first line is useless information
    accu_num = 0;
    for l in lines[1:]:
        chk_res = check_line(model, target_name, l)
        accu_num += chk_res
    
    accu_rate = 1.0 * accu_num / (len(lines) - 1)
    print("accuracy rate = {}".format(accu_rate))

if __name__ == "__main__":
    main()
