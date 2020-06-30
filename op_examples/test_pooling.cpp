#include "utilities.hpp"

migraphx::program sin_program()
{
    migraphx::program p;

    std::vector<float> data(4 * 5);
    std::iota(data.begin(), data.end(), 0);
    migraphx::shape s{migraphx::shape::float_type, {4, 5}};

    auto a0 = p.add_literal(migraphx::literal{s, data});
    p.add_instruction(migraphx::op::sin{}, a0);

    return p;
}

migraphx::program round_program()
{
    migraphx::program p;

    std::vector<float> data = {-3.5f, -2.5f, -1.5f, -0.5f, 0.0f, 0.5f, 1.5f, 2.5f, 3.5f};
    migraphx::shape s{migraphx::shape::float_type, {9}};

    auto a0 = p.add_literal(migraphx::literal{s, data});
    p.add_instruction(migraphx::op::round{}, a0);

    return p;
}


migraphx::program pooling_program() 
{
    migraphx::program p;
    auto input =
        p.add_parameter("x", migraphx::shape{migraphx::shape::float_type, {1, 3, 16, 16}});
    auto op    = migraphx::op::pooling{"average"};
    auto lens  = input->get_shape().lens();
    op.lengths = {lens[2], lens[3]};
    p.add_instruction(op, input);
    return p;
}

int main()
{
    auto p = pooling_program();
    std::vector<float> cpu_res, gpu_res;
    run_cpu(p, cpu_res);
    run_gpu(p, gpu_res);

    bool b_res = compare_results(cpu_res, gpu_res);

    std::cout << (b_res ? "PASSED!" : "FAILED!") << std::endl;

    return 0;
}
