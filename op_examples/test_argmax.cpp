#include <iostream>
#include <iomanip>
#include <vector>
#include <migraphx/program.hpp>
#include <migraphx/literal.hpp>
#include <migraphx/operators.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/cpu/target.hpp>
#include <migraphx/gpu/target.hpp>
#include <migraphx/gpu/hip.hpp>
#include <migraphx/manage_ptr.hpp>
#include <migraphx/type_name.hpp>
#include "test.hpp"
#include "utilities.hpp"

migraphx::program create_program()
{
    migraphx::program p;

    int axis = 0;
    //std::vector<float> data(3 * 4 * 5 * 6);
    //std::iota(data.begin(), data.end(), 0.5);
    migraphx::shape s{migraphx::shape::float_type, {128, 1, 1, 1}};
    //auto a0 = p.add_literal(migraphx::literal{s, data});
    auto a0 = p.add_parameter("a0", s);
    p.add_instruction(migraphx::op::argmax{axis}, a0);

    return p;
}

int main(int argc, char **argv) {
    std::vector<int64_t> cpu_res, gpu_res;
    migraphx::program prog = create_program();
    run_gpu(prog, gpu_res);
    run_cpu(prog, cpu_res);
    
    bool ret = compare_results(cpu_res, gpu_res);
    std::cout << (ret ? "PASSED!" : "FAILED!") << std::endl;

    return 0;
}

