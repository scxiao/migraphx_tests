#include <iostream>
#include <vector>
#include <string>
#include <migraphx/literal.hpp>
#include <migraphx/operators.hpp>
#include <migraphx/program.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/onnx.hpp>
#include "utilities.hpp"

migraphx::program load_onnx_file(std::string file_name) {
    auto prog = migraphx::parse_onnx(file_name);
    return prog;
}

int main(int argc, char **argv) {
    if (argc != 3) {
        std::cout << "Usage: " << argv[0] << " onnx_file cpu/gpu/both" << std::endl;
        return 0;
    }

    auto prog = load_onnx_file(argv[1]);
    int run_flag = 0;
    std::string str_flag(argv[2]);
    if (str_flag == "gpu")
    {
        run_flag = 1;
    }
    else if (str_flag == "both")
    {
        run_flag = 2;
    }

    std::vector<float> cpu_res, gpu_res;
    
    bool ret = true;
    if (run_flag == 0)
    {
        run_cpu(prog, cpu_res);
    }
    else if (run_flag == 1)
    {
        run_gpu(prog, gpu_res);
    }
    else 
    {
        run_cpu(prog, cpu_res);
        run_gpu(prog, gpu_res);
        ret = compare_results(cpu_res, gpu_res);
    }

    std::cout << (ret ? "PASSED!" : "FAILED") << std::endl;
    return 0;
}

