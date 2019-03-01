#include "utilities.hpp"

migraphx::program create_program()
{
    migraphx::program p;

    std::vector<float> data(4 * 5);
    std::iota(data.begin(), data.end(), 0);
    migraphx::shape s{migraphx::shape::float_type, {4, 5}};

    auto a0 = p.add_literal(migraphx::literal{s, data});
    p.add_instruction(migraphx::op::sin{}, a0);

    return p;
}

int main()
{
    auto p1 = create_program();
    auto p2 = create_program();
    std::vector<float> cpu_res, gpu_res;
    run_cpu(p1, cpu_res);
    run_gpu(p2, gpu_res);

    bool b_res = compare_results(cpu_res, gpu_res);

    std::cout << (b_res ? "PASSED!" : "FAILED!") << std::endl;

    return 0;
}
