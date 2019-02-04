#include <iostream>
#include <vector>
#include <migraphx/literal.hpp>
#include <migraphx/operators.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/cpu/target.hpp>
#include <migraphx/verify.hpp>
#include "test.hpp"

int main()
{
    migraphx::program p;

    std::vector<float> data(4 * 5);
    std::iota(data.begin(), data.end(), 0);
    migraphx::shape s{migraphx::shape::float_type, {4, 5}};

    auto a0 = p.add_literal(migraphx::literal{s, data});
    p.add_instruction(migraphx::op::sin{}, a0);
    p.compile(migraphx::cpu::target{});
    auto result = p.eval({});
    std::vector<float> resData(4 * 5);
    result.visit([&](auto output) { resData.assign(output.begin(), output.end()); });

    std::cout << "res = " << std::endl;
    for_each(resData.begin(), resData.end(), [](float& i) { std::cout << i << "\t"; });
    std::cout << std::endl;

    return 0;
}
