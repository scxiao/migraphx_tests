#include <iostream>
#include <vector>
#include <string>
#include "utilities.hpp"
#include <migraphx/literal.hpp>
#include <migraphx/operators.hpp>
#include <migraphx/program.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/gpu/target.hpp>
#include <migraphx/onnx.hpp>

migraphx::program load_onnx_file(std::string file_name) {
    auto prog = migraphx::parse_onnx(file_name);
    std::cout << "Load program is: " << std::endl;
    std::cout << prog << std::endl;
    prog.compile(migraphx::gpu::target{});
    
    return prog;
}

int main(int argc, char **argv) {
    if (argc != 2) {
        std::cout << "Usage: " << argv[0] << " onnx_file" << std::endl;
        return 0;
    }

    auto p = load_onnx_file(argv[1]);
    migraphx::program copy_p(p);

    std::cout << "copy_p = " << std::endl;
    std::cout << copy_p << std::endl;
    std::cout << "p = " << std::endl;
    std::cout << p << std::endl;

    bool comp_res = (copy_p == p);
    std::cout << "programs are " << (comp_res ? "the same" : "not the same") << std::endl;

    return 0;
}

