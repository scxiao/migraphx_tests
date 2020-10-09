#include <iostream>
#include <vector>
#include <migraphx/literal.hpp>
#include <migraphx/operators.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/quantization.hpp>
#include <migraphx/cpu/target.hpp>
#include <migraphx/quantization.hpp>
#include <migraphx/verify.hpp>
#include "utilities.hpp"

template<class T>
void print(std::string name, std::vector<T>& vec)
{
    std::cout << name << " = ";
    for (auto v : vec)
    {
        std::cout << v << "\t";
    }
    std::cout << std::endl;
}

migraphx::program ave_pool()
{
    migraphx::program p;
    auto input =
        p.add_parameter("x", migraphx::shape{migraphx::shape::float_type, {1, 1, 5}});
    auto op = migraphx::op::pooling{"max", {1}, {3}, {3}, true};
    p.add_instruction(op, input);
    return p;
}

migraphx::program relu_lrn()
{
    migraphx::program p;
    auto x = p.add_parameter("x", migraphx::shape{migraphx::shape::float_type, {1, 5, 2, 2}});
    //auto y = p.add_instruction(migraphx::op::relu{}, x);
    p.add_instruction(migraphx::op::lrn{0.0001, 0.75, 1.0, 5}, x);
    return p;
}

int main()
{
    auto p = relu_lrn();
    std::vector<std::vector<float>> cpu_result, gpu_result;
    run_prog(p, migraphx::cpu::target{}, cpu_result);
    run_prog(p, migraphx::gpu::target{}, gpu_result);

    print("cpu", cpu_result[0]);
    print("gpu", gpu_result[0]);
    
    //migraphx::shape output{migraphx::shape::float_type, {4, 3, 1, 1}};
    //migraphx::shape input{migraphx::shape::float_type, {4, 3, 3, 3}};
    //migraphx::program p;
    //auto l = p.add_parameter("a", input);
    //auto r = p.add_instruction(migraphx::op::pooling{"max", {0, 0}, {3, 3}, {1, 1}}, l);
    //auto r1 = p.add_instruction(migraphx::op::pooling{"max", {0, 0}, {3, 3}, {1, 1}, migraphx::op::default_, 1}, l);
    //p.add_return({r, r1});

    //std::cout << "p = " << p << std::endl;

    return 0;
}

