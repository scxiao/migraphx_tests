#include <iostream>
#include <vector>
#include <string>
#include <migraphx/literal.hpp>
#include <migraphx/quantization.hpp>
#include <migraphx/operators.hpp>
#include <migraphx/program.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/gpu/target.hpp>
#include <migraphx/onnx.hpp>
#include "utilities.hpp"

migraphx::program load_onnx_file(std::string file_name) {
    auto prog = migraphx::parse_onnx(file_name);
    return prog;
}


void print_arg(migraphx::argument& arg)
{
    std::vector<float> vec_val;
    arg.visit([&](auto output) { vec_val.assign(output.begin(), output.end()); });
    std::size_t num = 768;
    num = num > vec_val.size() ? vec_val.size(): num;
    for (size_t i = 0; i < num; ++i)
    {
        std::cout << std::setw(12) << vec_val[i];
        if ((i + 1) % 8 == 0)
            std::cout << std::endl;
    }
    std::cout << std::endl << std::endl;

    return;
}

size_t cpu_arg_index = 0;
size_t gpu_arg_index = 0;
// function to print input arguments
void print_params_cpu(std::size_t ins_index, std::vector<migraphx::argument> args)
{
    for (auto& arg : args)
    {
        std::cout << "arg_index = " << cpu_arg_index++ << std::endl;
        std::cout << "arg_shape = " << arg.get_shape() << std::endl;
        print_arg(arg);
    }
};

// function to print input arguments
void print_params_gpu(std::size_t ins_index, std::vector<migraphx::argument> args)
{
    for (auto& arg : args)
    {
        auto cpu_arg = migraphx::gpu::from_gpu(arg);
        std::cout << "arg_index = " << gpu_arg_index++ << std::endl;
        std::cout << "arg_shape = " << arg.get_shape() << std::endl;
        print_arg(cpu_arg);
    }
};


int main(int argc, char **argv) {
    if (argc != 3) {
        std::cout << "Usage: " << argv[0] << " onnx_file ins_index" << std::endl;
        return 0;
    }

    std::size_t ins_index = std::atol(argv[2]);
    std::cout << "capture_ins_index = " << ins_index << std::endl;

    auto cpu_prog = load_onnx_file(argv[1]);
    auto gpu_prog = cpu_prog;

    std::cout << "prog = " << std::endl;
    //std::cout << cpu_prog << std::endl;

    std::vector<float> cpu_res, gpu_res;
	capture_ins_arguments(cpu_prog, ins_index, print_params_cpu);
    run_cpu(cpu_prog, cpu_res);
	capture_ins_arguments(gpu_prog, ins_index, print_params_gpu);
    run_gpu(gpu_prog, gpu_res);
    bool ret2 = compare_results(cpu_res, gpu_res);
    std::cout << (ret2 ? "PASSED!" : "FAILED") << std::endl;

    return 0;
}

