import onnx
from onnx import version_converter, helper
from onnxruntime.quantization import quantize_qat, QuantType
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
#    print(onnx.helper.printable_graph(original_model))
    quantized_model = quantize_qat(model_path, out_file_path)
    print("Quantized model = \n")
    print(onnx.helper.printable_graph(quantized_model))

    #onnx.save(converted_model, out_file_path)

if __name__ == "__main__":
    main()
