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

template <class T>
auto get_hash(const T& x)
{
    return std::hash<T>{}(x);
}

int main()
{
    {
        migraphx::program p;

        std::vector<double> data(7 * 5);
        std::iota(data.begin(), data.end(), 0);
        migraphx::shape s_x{migraphx::shape::double_type, {1, 1, 7, 5}};
        auto x = p.add_literal(migraphx::literal{s_x, data});
        migraphx::shape s_k{migraphx::shape::double_type, {1, 1, 3, 3}};
        std::vector<double> weight(3 * 3, 1.0);
        auto w = p.add_literal(migraphx::literal{s_k, weight});
        p.add_instruction(migraphx::op::convolution{{1, 1}, {2, 2}}, x, w);
        p.compile(migraphx::cpu::target{});
        auto result = p.eval({});
        migraphx::shape result_shape = result.get_shape();
        std::cout << "Output shape: " << std::endl;
        std::cout << result_shape << std::endl;

        std::vector<double> resData(5 * 5);
        result.visit([&](auto output) { resData.assign(output.begin(), output.end()); });

        std::cout << "convoluation res = " << std::endl;
        for_each(resData.begin(), resData.end(), [](double& i) { std::cout << i << "\t"; });
        std::cout << std::endl;
    }

    {
        migraphx::program p;

        std::vector<double> data(7 * 5);
        std::iota(data.begin(), data.end(), 0);
        migraphx::shape s_x{migraphx::shape::double_type, {7, 5}};
        auto x = p.add_literal(migraphx::literal{s_x, data});

        migraphx::shape s_k{migraphx::shape::double_type, {5, 3}};
        std::vector<double> weight(5 * 3, 1.0);
        auto w = p.add_literal(migraphx::literal{s_k, weight});
        p.add_instruction(migraphx::op::dot{}, x, w);
        p.compile(migraphx::cpu::target{});
        auto result = p.eval({});
        migraphx::shape result_shape = result.get_shape();
        std::cout << "Output shape: " << std::endl;
        std::cout << result_shape << std::endl;

        std::vector<double> resData(5 * 5);
        result.visit([&](auto output) { resData.assign(output.begin(), output.end()); });

        std::cout << "matrix_mul res = " << std::endl;
        for_each(resData.begin(), resData.end(), [](double& i) { std::cout << i << "\t"; });
        std::cout << std::endl;
    }


    return 0;
}
