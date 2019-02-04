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
        migraphx::shape sa{migraphx::shape::int32_type, {1, 4, 5}};
        auto a = p.add_literal(migraphx::literal{sa, data});

        data.resize(5 * 6);
        std::iota(data.begin(), data.end(), 1);
        migraphx::shape sb{migraphx::shape::int32_type, {1, 5, 6}};
        auto b = p.add_literal(migraphx::literal{sb, data});

        auto ia = p.add_instruction(migraphx::op::dot{}, a, b);

        p.compile(migraphx::cpu::target{});
        auto result = p.eval({});
        std::vector<int> resData(4 * 5);
        result.visit([&](auto output) { resData.assign(output.begin(), output.end()); });

        std::cout << "res = " << std::endl;
        for_each(resData.begin(), resData.end(), [](int& i) { std::cout << i << "\t"; });
        std::cout << std::endl;
    }

    return 0;
}
