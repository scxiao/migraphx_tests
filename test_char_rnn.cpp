#include <iostream>
#include <vector>
#include <string>
#include <migraphx/literal.hpp>
#include <migraphx/operators.hpp>
#include <migraphx/program.hpp>
#include <migraphx/cpu/target.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/onnx.hpp>

migraphx::program load_onnx_file(std::string file_name) {
    auto prog = migraphx::parse_onnx(file_name);

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
    migraphx::shape hidden_shape{migraphx::shape::float_type, {1, 128}};
    migraphx::shape input_shape{migraphx::shape::float_type, {1, 57}};

    std::vector<float> hidden_state(hidden_size, 1.0);
    std::vector<float> input(input_size, 0.0);
    input[10] = 1;

    prog.compile(migraphx::cpu::target{});

    migraphx::program::parameter_map m;
    for (auto&& x : prog.get_parameter_shapes()) {
        std::cout << x.first << "'s shape:" << std::endl;
        std::cout << x.second << std::endl;
    }
    m["0"] = migraphx::argument(input_shape, &input[0]);
    std::cout << "m0 = " << m["0"] << std::endl;

    m["1"] = migraphx::argument(hidden_shape, &hidden_state[0]);
    std::cout << "m1 = " << m["1"] << std::endl;

    auto resarg = prog.eval(m);

    std::vector<float> res(input_size + hidden_size + 100);
    resarg.visit([&](auto output) { res.assign(output.begin(), output.end()); } );
    std::cout << "output size = " << res.size() << std::endl;
    for_each(res.begin(), res.end(), [](auto &i) { std::cout << i << "\t";} );
    std::cout << std::endl;

    std::cout << "prog = " << std::endl;
    std::cout << prog << std::endl;

    return 0;
}

