#include <iostream>
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

int main()
{
    migraphx::program p;
    migraphx::shape s{migraphx::shape::int32_type, {4, 4, 4}};
    std::vector<int> data(s.elements());
    std::iota(data.begin(), data.end(), 0);
    auto l0 = p.add_literal(migraphx::literal{s, data});
    p.add_instruction(migraphx::op::split{1, {1, 2, 1}}, l0);
    p.compile(migraphx::cpu::target{});
    auto result           = p.eval({});
    std::cout << "output_shape_0  =  " << result.get_shape() << std::endl;
    std::cout << "res = " << result << std::endl;

    return 0;
}
