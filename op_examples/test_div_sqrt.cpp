#include "utilities.hpp"

migraphx::program create_program()
{
    migraphx::program p;
    migraphx::shape a_shape{migraphx::shape::float_type, {4, 5}};
    std::vector<std::size_t> lens = {3, 4, 5};
    migraphx::shape b_shape{migraphx::shape::float_type, lens};
    auto ia = p.add_parameter("a", a_shape);
    auto ib = p.add_parameter("b", b_shape);

    auto i2 = p.add_instruction(migraphx::op::mul{}, ia, ia);
    auto i3 = p.add_instruction(migraphx::op::sqrt{}, i2);
    auto i4 = p.add_instruction(migraphx::op::multibroadcast{lens}, i3);
    p.add_instruction(migraphx::op::div{}, ib, i4);

    return p;
}

int main()
{
    auto p = create_program();
    std::vector<float> cpu_res, gpu_res;
    run_cpu(p, cpu_res);
    run_gpu(p, gpu_res);

    bool b_res = compare_results(cpu_res, gpu_res);
    std::cout << (b_res ? "PASSED!" : "FAILED") << std::endl;

    return 0;
}

