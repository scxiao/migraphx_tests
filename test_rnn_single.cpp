#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <migraphx/literal.hpp>
#include <migraphx/operators.hpp>
#include <migraphx/program.hpp>
#include <migraphx/cpu/target.hpp>
#include <migraphx/gpu/target.hpp>
#include <migraphx/gpu/hip.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/onnx.hpp>

migraphx::program load_onnx_file(std::string file_name) {
    auto prog = migraphx::parse_onnx(file_name);

    std::cout << "prog = " << std::endl;
    std::cout << prog << std::endl;

    return prog;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        std::cout << "Usage: " << argv[0] << " onnx_file gpu/cpu" << std::endl;
        return 0;
    }

    migraphx::program prog = load_onnx_file(argv[1]);

    bool b_use_gpu = false;
    if (argc == 3 && std::string(argv[2]) == std::string("gpu"))
    {
        b_use_gpu = true;
    }

    if (b_use_gpu) 
    {
        std::cout << "gpu is used." << std::endl;
        prog.compile(migraphx::gpu::target{});
    }
    else 
    {
        std::cout << "cpu is used." << std::endl;
        prog.compile(migraphx::cpu::target{});
    }

    migraphx::program::parameter_map m;
    std::cout << "Input of rnn is:-----------------------------------" << std::endl;
    for (auto&& x : prog.get_parameter_shapes()) {
        std::cout << x.first << "'s shape:" << std::endl;
        std::cout << x.second << std::endl;
    }
    std::vector<float> res;
    std::vector<std::vector<float>> data(prog.get_parameter_shapes().size());
    int index = 0;
    for (auto &&x : prog.get_parameter_shapes())
    {
        std::cout << "x.first.shape = " << x.first << std::endl;
        if (x.first == "input") {
            data[index].resize(x.second.elements(), 0.0f);
            data[index][0] = data[index][1] = 1.0;
        }
        else if (x.first == "1")
        {
            data[index].resize(x.second.elements(), 1.0f);
        }
        else
        {
            data[index].resize(x.second.elements(), 1.0f);
        }

        if (b_use_gpu) {
            m[x.first] = migraphx::gpu::to_gpu(migraphx::argument{x.second, data[index++].data()});
        }
        else {
            m[x.first] = migraphx::argument{x.second, data[index++].data()};
        }
    }

    if (b_use_gpu) {
        auto resarg = migraphx::gpu::from_gpu(prog.eval(m));
        resarg.visit([&](auto output) { res.assign(output.begin(), output.end()); } );
        std::cout << "gpu result: " << std::endl;
    }
    else {
        auto resarg = prog.eval(m);
        resarg.visit([&](auto output) { res.assign(output.begin(), output.end()); } );
        std::cout << "cpu result: " << std::endl;
    }


    std::cout << "output size = " << res.size() << std::endl;
    for (size_t i = 0; i < res.size(); i++) {
        std::cout << std::setw(12) << res.at(i) <<",";
        if ((i + 1) % 5 == 0) {
            std::cout << std::endl;
        }
    }
    std::cout << std::endl;

    return 0;
}

