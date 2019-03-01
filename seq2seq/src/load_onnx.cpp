#include "load_onnx.hpp"

migraphx::program load_onnx_file(std::string file_name) {
    auto prog = migraphx::parse_onnx(file_name);
    return prog;
}


