#include <iostream>
#include <vector>
#include <string>
#include <migraphx/literal.hpp>
#include <migraphx/operators.hpp>
#include <migraphx/program.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/gpu/target.hpp>
#include <migraphx/onnx.hpp>
#include "utilities.hpp"


migraphx::program create_program()
{ 
    migraphx::program p;

    migraphx::shape s{migraphx::shape::float_type, {30522, 768}};
    std::vector<float> data(s.elements());
    std::iota(data.begin(), data.end(), 0.5);
    auto a0 = p.add_literal(migraphx::literal{s, data});
    migraphx::shape s_indices{migraphx::shape::int32_type, {1, 11}};
    std::vector<int> indices(s_indices.elements(), 1);
    auto a1 = p.add_literal(migraphx::literal{s_indices, indices});
    auto g = p.add_instruction(migraphx::op::gather{0}, a0, a1);

    return p;
}

int main()
{

    auto prog = create_program();
    std::vector<float> cpu_res, gpu_res;
    run_cpu(prog, cpu_res);
    run_gpu(prog, gpu_res);
    bool ret2 = compare_results(cpu_res, gpu_res);
    std::cout << (ret2 ? "PASSED!" : "FAILED") << std::endl;

    return 0;
}
