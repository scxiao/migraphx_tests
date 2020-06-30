import onnx
from onnx import version_converter, helper
import sys

def main():
    if len(sys.argv) != 3:
        print("usage: python convert_version.py [onnx_file] [output_file]")
        exit()

    model_path = sys.argv[1]
    out_file_path = sys.argv[2]
    original_model = onnx.load(model_path)
    onnx.checker.check_model(original_model)

    print("Original graph = \n")
    print(onnx.helper.printable_graph(original_model))
    converted_model = version_converter.convert_version(original_model, 7)

    print("Converted graph = \n")
    print(onnx.helper.printable_graph(converted_model))

    onnx.save(converted_model, out_file_path)
