#include "../mytests/utilities.hpp"

migraphx::program create_program() {
    migraphx::program p1;
    migraphx::shape s{migraphx::shape::float_type, {128, 96}};
    auto input = p1.add_parameter("input", s);
    auto slc0  = p1.add_instruction(migraphx::op::slice{{1}, {0}, {32}}, input);
    auto slc1  = p1.add_instruction(migraphx::op::slice{{1}, {32}, {64}}, input);
    auto slc2  = p1.add_instruction(migraphx::op::slice{{1}, {64}, {96}}, input);

    auto c0 = p1.add_instruction(migraphx::op::contiguous{}, slc0);
    auto c1 = p1.add_instruction(migraphx::op::contiguous{}, slc1);
    auto c2 = p1.add_instruction(migraphx::op::contiguous{}, slc2);

    std::vector<int64_t> lens = {1, 16, 8, 32};
    auto r0                   = p1.add_instruction(migraphx::op::reshape{lens}, c0);
    auto r1                   = p1.add_instruction(migraphx::op::reshape{lens}, c1);
    auto r2                   = p1.add_instruction(migraphx::op::reshape{lens}, c2);

    auto sum = p1.add_instruction(migraphx::op::add{}, r0, r1);
    auto ret = p1.add_instruction(migraphx::op::mul{}, sum, r2);
    p1.add_return({ret});

    return p1;
};


int main()
{
    auto p = create_program();
    std::cout << "p = " << std::endl;
    std::cout << p << std::endl;
    p.compile(migraphx::gpu::target{});
    std::cout << "p = " << std::endl;
    std::cout << p << std::endl;

    return 0;
}
