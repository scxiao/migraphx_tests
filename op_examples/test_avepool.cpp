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

migraphx::program load_onnx_file(std::string file_name) {
    auto prog = migraphx::parse_onnx(file_name);
    std::cout << "Load program is: " << std::endl;
    std::cout << prog << std::endl;

    return prog;
}

void print_res(std::vector<float>& res, int m, int n)
{
    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < n; j++)
        {
            std::cout << "\t" << res[i * n + j];
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

int main(int argc, char **argv) {
    if (argc != 2) {
        std::cout << "Usage: " << argv[0] << " onnx_file" << std::endl;
        return 0;
    }

    auto prog = migraphx::parse_onnx(std::string(argv[1]));
    prog.compile(migraphx::cpu::target{});

    auto in_shape = prog.get_parameter_shapes()["x"];
    std::vector<float> vec(in_shape.elements());
    std::iota(vec.begin(), vec.end(), 1.0f);

    migraphx::program::parameter_map m;
    m["x"] = migraphx::argument(in_shape, vec.data());

    auto arg_result = prog.eval(m).back();
    std::vector<float> result;
    arg_result.visit([&](auto out) { result.assign(out.begin(), out.end()); });
    auto out_lens = arg_result.get_shape().lens();
    print_res(result, out_lens[2], out_lens[3]);

    return 0;
}

