#include <iostream>
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
#include "utilities.hpp"

migraphx::program create_program()
{
    migraphx::program p;
    std::vector<float> a = {0.7481789 ,  0.02906279,  1.01193836,  1.60222907,  1.89135978,
        0.30054158, -0.4892588 , -0.27027533};
    std::vector<float> b = {-0.25829116,  0.27908929, -1.27888957,  0.21152361,  0.08593658,
        0.52163899,  1.38343824, -0.2342857};
    migraphx::shape a_shape{migraphx::shape::float_type, {8}};
    auto al = p.add_literal(migraphx::literal{a_shape, a});
    migraphx::shape b_shape{migraphx::shape::float_type, {8}};
    auto bl = p.add_literal(migraphx::literal{b_shape, b});
    auto sl = p.add_instruction(migraphx::op::add{}, al, bl);
    auto s1l = p.add_instruction(migraphx::op::add{}, sl, al);
    auto s2l = p.add_instruction(migraphx::op::add{}, sl, bl);
    p.add_instruction(migraphx::op::add{}, s1l, s2l);


    return p;
}

migraphx::program create_program1()
{
    migraphx::program p;
    std::vector<float> a = {0.7481789 ,  0.02906279,  1.01193836,  1.60222907,  1.89135978,
        0.30054158, -0.4892588 , -0.27027533};
    std::vector<float> b = {-0.25829116,  0.27908929, -1.27888957,  0.21152361,  0.08593658,
        0.52163899,  1.38343824, -0.2342857};
    migraphx::shape a_shape{migraphx::shape::float_type, {8}};
    auto al = p.add_literal(migraphx::literal{a_shape, a});
    migraphx::shape b_shape{migraphx::shape::float_type, {8}};
    auto bl = p.add_parameter("b", b_shape);
    auto sl = p.add_instruction(migraphx::op::add{}, al, bl);
    auto s1l = p.add_instruction(migraphx::op::add{}, sl, al);
    auto s2l = p.add_instruction(migraphx::op::add{}, sl, bl);
    p.add_instruction(migraphx::op::add{}, s1l, s2l);


    return p;
}


int main(int argc, char **argv) {
    std::vector<float> cpu_res, gpu_res;
    auto p1 = create_program1();
    std::cout << "p1 = \n" << p1 << std::endl;
    migraphx::quantize(p1);
    run_cpu(p1, cpu_res);

    auto p2 = create_program1();
    migraphx::quantize(p2);
    run_gpu(p2, gpu_res);
    
    bool res = compare_results(cpu_res, gpu_res);
    std::cout << (res ? "PASSED" : "FAILED") << std::endl;

    return 0;
}

