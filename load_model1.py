import sys
import onnx
from onnx import helper

def main():
    file_name = sys.argv[1]
    print(file_name)
    model=onnx.load(file_name)
    onnx.checker.check_model(model)
    print(onnx.helper.printable_graph(model.graph))
    model1 = helper.make_model(model.graph, producer_name = 'tmp.onnx')
    onnx.save(model1, "tmp.onnx")


if __name__ == "__main__":
    main()

