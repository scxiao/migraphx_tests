#include <iostream>
#include <vector>
#include <string>
#include <migraphx/literal.hpp>
#include <migraphx/operators.hpp>
#include <migraphx/quantization.hpp>
#include <migraphx/pass_manager.hpp>
#include <migraphx/program.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/gpu/target.hpp>
#include <migraphx/cpu/target.hpp>
#include <migraphx/onnx.hpp>
#include <migraphx/rewrite_rnn.hpp>
#include <migraphx/dead_code_elimination.hpp>
#include <migraphx/pass_manager.hpp>
#include "utilities.hpp"

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

    std::string file_name(argv[1]);
    auto prog = load_onnx_file(file_name);

    std::vector<migraphx::pass> vec_passes = {
        migraphx::dead_code_elimination{},
        migraphx::rewrite_rnn{},
        migraphx::dead_code_elimination{}
    };

    migraphx::run_passes(prog, vec_passes, migraphx::tracer{std::cout});

    std::vector<float> cpu_res, gpu_res;
    //std::cout << "p1 = \n" << p1 << std::endl;
    //run_cpu(p1, cpu_res);

    //auto p2 = create_program1();
    //run_gpu(p2, gpu_res);
    
    //bool res = compare_results(cpu_res, gpu_res);
    //std::cout << (res ? "PASSED" : "FAILED") << std::endl;

    run_gpu(prog, gpu_res);

    return 0;
}

