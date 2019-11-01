#include "utilities.hpp"

migraphx::program create_program()
{
    migraphx::program p;
    migraphx::shape s1{migraphx::shape::float_type, {4, 5}};
    migraphx::shape s2{migraphx::shape::float_type, {1}};
    auto a1 = p.add_parameter("1", s1);
    auto a2 = p.add_parameter("2", s2);
    auto ba2 = p.add_instruction(migraphx::op::multibroadcast{s1.lens()}, a2);

    p.add_instruction(migraphx::op::add{}, ba2, a1);

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
