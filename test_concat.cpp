#include <iostream>
#include <vector>
#include <string>
#include <migraphx/literal.hpp>
#include <migraphx/operators.hpp>
#include <migraphx/program.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/onnx.hpp>

migraphx::program load_onnx_file(std::string file_name) {
    auto prog = migraphx::parse_onnx(file_name);

    std::cout << "prog = " << std::endl;
    std::cout << prog << std::endl;

    return prog;
}


int main(int argc, char **argv) {
    if (argc != 2) {
        std::cout << "Usage: " << argv[0] << " onnx_file" << std::endl;
        return 0;
    }

    migraphx::program prog = load_onnx_file(argv[1]);
    size_t hidden_size = 128;
    size_t input_size = 57;
    migraphx::shape hidden_shape{migraphx::shape::double_type, {1, 128}};
    migraphx::shape input_shape{migraphx::shape::double_type, {1, 57}};

    std::vector<double> hidden_state(hidden_size, 0.0);
    std::vector<double> input(input_size, 0.0);
    input[10] = 1;

    migraphx::program::parameter_map m;
    for (auto&& x : prog.get_parameter_shapes()) {
        std::cout << x.first << "'s shape:" << std::endl;
        std::cout << x.second << std::endl;
    }
    m["1"] = migraphx::argument(input_shape, &input);
    m["0"] = migraphx::argument(hidden_shape, &hidden_state);

    auto resarg = prog.eval(m);



    return 0;
}

