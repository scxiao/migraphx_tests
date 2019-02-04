#include <iostream>
#include <vector>
#include <string>
#include <migraphx/literal.hpp>
#include <migraphx/operators.hpp>
#include <migraphx/program.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/onnx.hpp>

void load_onnx_file(std::string file_name) {
    auto prog = migraphx::parse_onnx(file_name);
    std::cout << "Load program is: " << std::endl;
    std::cout << prog << std::endl;
}

int main(int argc, char **argv) {
    if (argc != 2) {
        std::cout << "Usage: " << argv[0] << " onnx_file" << std::endl;
        return 0;
    }

    load_onnx_file(argv[1]);

    return 0;
}

