import onnx
import os
import sys
import onnxmltools
import argparse
from onnxmltools.utils.float16_converter import convert_float_to_float16

def convert_to_fp16(model):
    onnx_model = convert_float_to_float16(model)
    return onnx_model

def parse_args():
    parser = argparse.ArgumentParser(
        os.path.basename(__file__),
        description='Convert model from float32 to float16.')


    parser.add_argument('--input', required=True, help='The input model file')
    parser.add_argument('--output', help='The output model file')

    return parser.parse_args()

def main():
    args = parse_args()
    outfile = args.output
    infile= args.input
    filename, ext = os.path.splitext(infile)
    if args.output is None:
        outfile=filename + '_fp16' + ext

    print("input_model = {}".format(infile))
    print("output_model = {}".format(outfile))
        
    input_model = onnxmltools.utils.load_model(infile)
    model_fp16 = convert_to_fp16(input_model)
    onnxmltools.utils.save_model(model_fp16, outfile)

if __name__ == '__main__':
    main()
