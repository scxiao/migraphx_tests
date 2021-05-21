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
#include <migraphx/ref/target.hpp>
#include <migraphx/onnx.hpp>
#include "read_shape.hpp"
#include "cmd_options.hpp"

void print_param_shapes(const migraphx::program& p)
{
    std::cout << "============parameter names and shapes============" << std::endl;
    auto param_shapes = p.get_parameter_shapes();
    for (auto& pm : param_shapes)
    {
        std::cout << "name = " << pm.first << ", shape = " << pm.second << std::endl;
    }
    std::cout << std::endl;
}

void print_output_shapes(const migraphx::program& p)
{
    std::cout << "********** output shapes **************" << std::endl;
    auto out_shapes = p.get_output_shapes();
    std::size_t i = 0;
    for (auto& s : out_shapes)
    {
        std::cout << "#output_" << i++ << " shape: " << s << std::endl;
    }
    std::cout << std::endl;
}

void load_onnx_file(std::string file_name, migraphx::onnx_options options)
{
    options.print_program_on_error = true;
    auto prog = migraphx::parse_onnx(file_name, options);
    std::cout << "Load program is: " << std::endl;
    std::cout << prog << std::endl;
    std::cout << std::endl;

    auto p1 = prog;
    std::cout << "p1 = " << std::endl;
    std::cout << p1;
    print_param_shapes(p1);
    print_output_shapes(p1);
    

    p1.compile(migraphx::ref::target{});
    std::cout << "After compiling on REF, program is: " << std::endl;
    std::cout << p1;
    print_param_shapes(p1);
    print_output_shapes(p1);

    prog.compile(migraphx::gpu::target{});

    std::cout << "After compiling on GPU, program is: " << std::endl;
    std::cout << prog;
    print_param_shapes(prog);
    print_output_shapes(prog);
}

int main(int argc, char **argv) {
    if (argc < 2) {
        std::cout << "Usage: " << argv[0] << " onnx_file" << std::endl;
        std::cout << "\t-options   input_shape_file" << std::endl;
        return 0;
    }

    migraphx::onnx_options options;
    char *option_input_file = getCmdOption(argv + 2, argv + argc, "-options");
    if (option_input_file != nullptr)
    {
        options = load_option_file(std::string(option_input_file));
    }

    load_onnx_file(argv[1], options);

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

