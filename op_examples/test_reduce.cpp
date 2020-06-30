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

migraphx::program create_program_argmax()
{
    migraphx::program p;

    int64_t axis = -1;
    //std::vector<float> data(3 * 4 * 5 * 6);
    //std::iota(data.begin(), data.end(), 0.5);
    migraphx::shape s{migraphx::shape::float_type, {128, 1, 1, 1}};
    //auto a0 = p.add_literal(migraphx::literal{s, data});
    auto a0 = p.add_parameter("a0", s);
    p.add_instruction(migraphx::op::argmax{axis}, a0);

    return p;
}

migraphx::program create_program()
{
    migraphx::program p;
    migraphx::shape s{migraphx::shape::int32_type, {3, 4, 8, 8}};
    std::vector<int> data(s.elements());
    std::iota(data.begin(), data.end(), 0);
    auto x = p.add_literal(migraphx::literal(s, data));
    int64_t axis = -1;
    p.add_instruction(migraphx::op::reduce_sum{{axis}}, x);
    return p;
};

migraphx::program create_program_1()
{
    migraphx::program p;
    migraphx::shape s{migraphx::shape::float_type, {4, 2, 2}};
    auto input = migraphx::literal{s, {1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 3, 2, 3}};
    auto l0    = p.add_literal(input);
    p.add_instruction(migraphx::op::reduce_log_sum_exp{{-1}}, l0);

	return p;
}

migraphx::program create_program_exp()
{
    migraphx::program p;
    migraphx::shape s{migraphx::shape::float_type, {3, 4, 8, 8}};
    auto x = p.add_parameter("x", s);
    auto ax = p.add_instruction(migraphx::op::clip{2.0f, -2.0f}, x);
    p.add_instruction(migraphx::op::reduce_log_sum_exp{{0, 1}}, ax);
    return p;
}

int main(int argc, char **argv) {
    std::vector<float> cpu_res, gpu_res;
    migraphx::program prog = create_program_1();
    run_gpu(prog, gpu_res);
    run_cpu(prog, cpu_res);
    
    bool ret = compare_results(cpu_res, gpu_res);
    std::cout << (ret ? "PASSED!" : "FAILED!") << std::endl;

    return 0;
}

