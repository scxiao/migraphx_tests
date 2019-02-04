#include <iostream>
#include <vector>
#include <migraphx/literal.hpp>
#include <migraphx/operators.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/argument.hpp>
#include <migraphx/cpu/target.hpp>
#include <migraphx/verify.hpp>
#include "test.hpp"

int main()
{
    {
        migraphx::program p;
        std::vector<int> data(4 * 5);
        std::iota(data.begin(), data.end(), 0);
        migraphx::shape s{migraphx::shape::int32_type, {4, 5}};
        auto a0 = p.add_literal(migraphx::literal{s, data});
        migraphx::argument arg0 = a0->eval();
        std::cout << "a0's shape is: " << arg0.get_shape() << std::endl;
        std::cout << "a0 is: " << arg0 << std::endl;

        std::iota(data.begin(), data.end(), 1);
        auto a1 = p.add_literal(migraphx::literal{s, data});
        migraphx::argument arg1 = a1->eval();
        std::cout << "a1's shape is: " << arg1.get_shape() << std::endl;

        auto ia = p.add_instruction(migraphx::op::add{}, a0, a1);
        migraphx::argument arg_sum = ia->eval();
        std::cout << "sum's shape is: " << arg_sum.get_shape() << std::endl;

        p.compile(migraphx::cpu::target{});
        auto result = p.eval({});
        std::vector<int> resData(4 * 5);
        result.visit([&](auto output) { resData.assign(output.begin(), output.end()); });

        std::cout << "res = " << std::endl;
        for_each(resData.begin(), resData.end(), [](int& i) { std::cout << i << "\t"; });
        std::cout << std::endl;
    }

    {
        migraphx::program p;
        std::vector<int> data(2 * 4 * 4 *5);
        std::iota(data.begin(), data.end(), 0);
        migraphx::shape s{migraphx::shape::int32_type, {2, 4, 4, 5}};
        auto a0 = p.add_literal(migraphx::literal{s, data});
        p.add_instruction(migraphx::op::slice{{0, 1}, {0, 0}, {1, 2}}, a0);
        p.compile(migraphx::cpu::target{});
        auto result_arg = p.eval({});
        std::vector<int> res;
        result_arg.visit([&](auto output) { res.assign(output.begin(), output.end()); });

        std::cout << "slice output shape = " << result_arg.get_shape() << std::endl;
    }

    {
        migraphx::program p;
        std::vector<int> data(2 * 4 * 4 *5);
        std::iota(data.begin(), data.end(), 0);
        migraphx::shape s{migraphx::shape::int32_type, {2, 4, 4, 5}};
        auto l0 = p.add_literal(migraphx::literal{s, data});
        p.add_instruction(migraphx::op::concat{1}, l0, l0);
        p.compile(migraphx::cpu::target{});
        auto result_arg = p.eval({});
        std::vector<int> res;
        result_arg.visit([&](auto output) { res.assign(output.begin(), output.end()); });

        std::cout << "concat output shape = " << result_arg.get_shape() << std::endl;
    }


    return 0;
}
