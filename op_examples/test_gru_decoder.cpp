#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <migraphx/literal.hpp>
#include <migraphx/operators.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/program.hpp>
#include <migraphx/cpu/target.hpp>
#include <migraphx/gpu/target.hpp>
#include <migraphx/gpu/hip.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/onnx.hpp>
#include "utilities.hpp"

migraphx::program load_onnx_file(std::string file_name) {
    auto prog = migraphx::parse_onnx(file_name);

    std::cout << "prog = " << std::endl;
    std::cout << prog << std::endl;

    return prog;
}

template<typename T>
void print_res(std::vector<T> &res)
{
    std::cout << "output size = " << res.size() << std::endl;
    for (size_t i = 0; i < res.size(); i++) {
        std::cout << std::setw(12) << res.at(i) <<",";

        if ((i + 1) % 8 == 0) {
            std::cout << std::endl;
        }
    }
}


void run_cpu(migraphx::program &p, std::vector<float> &res)
{
    p.compile(migraphx::cpu::target{});

    std::vector<long> decoder_input{3};
    std::vector<float> decoder_hidden(256, 1.0);
    std::vector<float> encoder_outputs(2560, 1.0);

    migraphx::program::parameter_map m;
    for (auto &&x : p.get_parameter_shapes())
    {
        if (x.first == "input.1")
        {
            std::cout << x.first << " shape = " << x.second << std::endl;
            m[x.first] = migraphx::argument{x.second, decoder_input.data()};
        }
        else if (x.first == "hidden")
        {
            std::cout << x.first << " shape = " << x.second << std::endl;
            m[x.first] = migraphx::argument{x.second, decoder_hidden.data()};
        }
        else if (x.first == "2")
        {
            std::cout << x.first << " shape = " << x.second << std::endl;
            m[x.first] = migraphx::argument{x.second, encoder_outputs.data()};
        }
        else
        {
            std::cout << x.first << " shape = " << x.second << std::endl;
            m[x.first] = migraphx::generate_argument(x.second, get_hash(x.first));
        }
    }

    auto result = p.eval(m);
    result.visit([&](auto output) { res.assign(output.begin(), output.end()); });

    std::cout << "cpu res = " << std::endl;
    print_res(res);
    std::cout << std::endl;
}

void run_gpu(migraphx::program &p, std::vector<float> &res)
{
    p.compile(migraphx::gpu::target{});

    std::vector<long> decoder_input{3};
    std::vector<float> decoder_hidden(256, 1.0);
    std::vector<float> encoder_outputs(2560, 1.0);

    migraphx::program::parameter_map m;
    for (auto &&x : p.get_parameter_shapes())
    {
        if (x.first == "input.1")
        {
            std::cout << x.first << " shape = " << x.second << std::endl;
            m[x.first] = migraphx::gpu::to_gpu(migraphx::argument{x.second, decoder_input.data()});
        }
        else if (x.first == "hidden")
        {
            std::cout << x.first << " shape = " << x.second << std::endl;
            m[x.first] = migraphx::gpu::to_gpu(migraphx::argument{x.second, decoder_hidden.data()});
        }
        else if (x.first == "2")
        {
            std::cout << x.first << " shape = " << x.second << std::endl;
            m[x.first] = migraphx::gpu::to_gpu(migraphx::argument{x.second, encoder_outputs.data()});
        }
        else
        {
            std::cout << x.first << " shape = " << x.second << std::endl;
            m[x.first] = migraphx::gpu::to_gpu(migraphx::generate_argument(x.second, get_hash(x.first)));
        }

    }
    auto result = migraphx::gpu::from_gpu(p.eval(m));
    result.visit([&](auto output) { res.assign(output.begin(), output.end()); });

    std::cout << "gpu res = " << std::endl;
    print_res(res);
    std::cout << std::endl;
}


int main(int argc, char **argv) {
    if (argc < 2) {
        std::cout << "Usage: " << argv[0] << " onnx_file gpu/cpu" << std::endl;
        return 0;
    }

    std::vector<float> cpu_res, gpu_res;
    migraphx::program prog2 = load_onnx_file(argv[1]);
    run_gpu(prog2, gpu_res);
    migraphx::program prog1 = load_onnx_file(argv[1]);
    run_cpu(prog1, cpu_res);

    std::size_t cpu_size = cpu_res.size();
    std::size_t gpu_size = gpu_res.size();
    if (cpu_size != gpu_size) {
        std::cout << "output size mistach!!!!!!!!!!!!!!!!" << std::endl;
    }

    bool passed = true;
    for (std::size_t i = 0; i < cpu_size; i++) {
        if (fabs(cpu_res[i] - gpu_res[i]) > 1.0e-6)
        {
            std::cout << "cpu_result[" << i << "] (" << cpu_res[i] << ") != gpu_result[" << i << "] (" <<
                gpu_res[i] << ")!!!!!!" << std::endl;
            passed = false;
        }
    }
    std::cout << (passed ? "PASSED!!!" : "FAILED!!!") << std::endl;

    return 0;
}


