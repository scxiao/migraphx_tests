#include <iostream>
#include <iomanip>
#include <vector>
#include <migraphx/program.hpp>
#include <migraphx/shape.hpp>
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
    migraphx::shape s{migraphx::shape::half_type, {1024, 4, 4096, 6}};
    auto param = p.add_parameter("0", s);
    p.add_instruction(migraphx::op::softmax{2}, param);

    return p;
}

//migraphx::program create_program()
//{
//    migraphx::program p;
//
//    int axis = 2;
//    //std::vector<float> data(3 * 4 * 5 * 6);
//    //std::iota(data.begin(), data.end(), 0.5);
//    migraphx::shape s{migraphx::shape::half_type, {1024, 4, 2080, 6}};
//    //auto a0 = p.add_literal(migraphx::literal{s, data});
//    auto a0 = p.add_parameter("a0", s);
//    p.add_instruction(migraphx::op::softmax{axis}, a0);
//
//    return p;
//}

int main(int argc, char **argv) {
    std::vector<float> cpu_res, gpu_res;
    migraphx::program prog = create_program();
    run_gpu(prog, gpu_res);
    run_cpu(prog, cpu_res);
    
    bool ret = compare_results(cpu_res, gpu_res);
    std::cout << (ret ? "PASSED!" : "FAILED!") << std::endl;

    return 0;
}

