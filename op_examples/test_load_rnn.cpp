#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <migraphx/literal.hpp>
#include <migraphx/quantization.hpp>
#include <migraphx/operators.hpp>
#include <migraphx/program.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/gpu/target.hpp>
#include <migraphx/cpu/target.hpp>
#include <migraphx/onnx.hpp>

void load_onnx_file(std::string file_name) {
    migraphx::onnx_options options;
    std::vector<std::size_t> vec{4, 128};
    options.map_input_dims["input_ids"] = vec;
    options.map_input_dims["input_mask"] = vec;
    options.map_input_dims["segment_ids"] = vec;
    auto prog = migraphx::parse_onnx(file_name, options);
    std::cout << "Load program is: " << std::endl;
    std::cout << prog << std::endl;
    //migraphx::quantize_int8(prog, migraphx::gpu::target{}, {});
    //std::cout << "Quantized program is: " << std::endl;
    //std::cout << prog << std::endl;
    //prog.compile(migraphx::gpu::target{});

    //std::cout << "After compiling, program is: " << std::endl;
    //std::cout << prog << std::endl;
}

//void load_onnx_string(std::string& model_str, std::vector<std::string>& unsupported_nodes) {
//    auto prog = migraphx::parse_model(model_str, unsupported_nodes);
//    std::cout << "Load program is: " << std::endl;
//    std::cout << prog << std::endl;
//    prog.compile(migraphx::gpu::target{});
//}


int main(int argc, char **argv) {
    if (argc != 2) {
        std::cout << "Usage: " << argv[0] << " onnx_file" << std::endl;
        return 0;
    }

    load_onnx_file(argv[1]);

    //std::ifstream ifs(std::string(argv[1]), std::ifstream::binary);
    //std::string model_str;
    //ifs.seekg(0, ifs.end);
    //std::size_t size = ifs.tellg();
    //ifs.seekg(0, ifs.beg);
    //model_str.resize(size);
    //ifs.read((char*)model_str.c_str(), size);

    //std::vector<std::string> unsupported_nodes;
    //load_onnx_string(model_str, unsupported_nodes);

    //std::cout << "Unsupported nodes:" << std::endl;
    //for (auto& node_name : unsupported_nodes)
    //{
    //    std::cout << node_name << "\t";
    //}
    //std::cout << std::endl << std::endl;

    return 0;
}

