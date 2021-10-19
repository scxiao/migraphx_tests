import sys, os
import numpy as np
import argparse
import onnx
from onnx import numpy_helper

def parse_args():
    parser = argparse.ArgumentParser(description="Read protobuf file")
    parser.add_argument('file_name',
                        type=str,
                        metavar='file_name',
                        help='protobuf data file name')
    args = parser.parse_args()

    return args


def read_pb_file(filename):
    try:
        pfile = open(filename, 'rb')
    except IOError:
        print("File {} open error".format(filename))
        sys.exit(1)

    data_str = pfile.read()
    tensor = onnx.TensorProto()
    tensor.ParseFromString(data_str)
    np_array = numpy_helper.to_array(tensor)
    print("data_shape = {}".format(np_array.shape))
    print("data = {}".format(np_array))

    return np_array


def main():
    args = parse_args()
    file_name = args.file_name

    read_pb_file(file_name)


if __name__ == "__main__":
    main()
