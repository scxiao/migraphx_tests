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

migraphx::program asinh_program()
{
    migraphx::program p;
    migraphx::shape s{migraphx::shape::float_type, {3}};
    std::vector<float> data{-0.5f, 0.0f, 0.9f};
    auto l = p.add_literal(migraphx::literal{s, data});
    p.add_instruction(migraphx::op::asinh{}, l);

    return p;
}


migraphx::program atanh_program()
{
    migraphx::program p;
    migraphx::shape s{migraphx::shape::float_type, {3}};
    auto l = p.add_literal(migraphx::literal{s, {0.4435683, 0.6223626, 0.316958}});
    p.add_instruction(migraphx::op::atanh{}, l);

	return p;
}

int main()
{
    auto p = atanh_program();
    std::vector<float> cpu_res, gpu_res;
    run_cpu(p, cpu_res);
    run_gpu(p, gpu_res);

    bool b_res = compare_results(cpu_res, gpu_res);

    std::cout << (b_res ? "PASSED!" : "FAILED!") << std::endl;

    return 0;
}
