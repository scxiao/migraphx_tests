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
    std::size_t axis = 1;
    //{
    //    migraphx::program p;

    //    std::vector<float> data(3 * 3);
    //    std::iota(data.begin(), data.end(), 0.5);
    //    migraphx::shape s{migraphx::shape::float_type, {3, 3}};
    //    auto a0 = p.add_literal(migraphx::literal{s, data});
    //    migraphx::shape s_indices{migraphx::shape::int32_type, {1, 2}};
    //    std::vector<int> indices{0, 2};
    //    auto a1 = p.add_literal(migraphx::literal{s_indices, indices});
    //    auto g = p.add_instruction(migraphx::op::gather{axis}, a0, a1);
    //    //migraphx::argument gr = g->eval();
    //    //std::cout << "gather output = " << gr << std::endl;

    //    p.compile(migraphx::cpu::target{});
    //    auto result = p.eval({});
    //    std::vector<float> resData(4 * 5);
    //    result.visit([&](auto output) { resData.assign(output.begin(), output.end()); });

    //    std::cout << "res = " << std::endl;
    //    for_each(resData.begin(), resData.end(), [](float & i) { std::cout << i << "\t"; });
    //    std::cout << std::endl;
    //}

    {
        migraphx::program p;

        std::vector<float> data{1, 2, 3, 4};
        //std::iota(data.begin(), data.end(), 0.5);
        migraphx::shape s{migraphx::shape::float_type, {2, 2}};
        auto a0 = p.add_literal(migraphx::literal{s, data});
        migraphx::shape s_indices{migraphx::shape::int32_type, {2, 2}};
        std::vector<int> indices{0, 0, 1, 0};
        auto a1 = p.add_literal(migraphx::literal{s_indices, indices});
        auto g = p.add_instruction(migraphx::op::gather_torch{axis}, a0, a1);
        //migraphx::argument gr = g->eval();
        //std::cout << "gather output = " << gr << std::endl;

        p.compile(migraphx::cpu::target{});
        auto result = p.eval({});
        std::vector<float> resData(4 * 5);
        result.visit([&](auto output) { resData.assign(output.begin(), output.end()); });

        std::cout << "res = " << std::endl;
        for_each(resData.begin(), resData.end(), [](float & i) { std::cout << i << "\t"; });
        std::cout << std::endl;
    }


    //{
    //    migraphx::program p;

    //    std::vector<float> data(3 * 3);
    //    std::iota(data.begin(), data.end(), 0.5);
    //    migraphx::shape s{migraphx::shape::float_type, {3, 3}};
    //    auto a0 = p.add_literal(migraphx::literal{s, data});
    //    migraphx::shape s_indices{migraphx::shape::int32_type, {1, 2}};
    //    std::vector<int> indices{0, 2};
    //    auto a1 = p.add_literal(migraphx::literal{s_indices, indices});
    //    p.add_instruction(migraphx::op::gather{axis}, a0, a1);
    //    p.compile(migraphx::gpu::target{});

    //    migraphx::program::parameter_map m;
    //    for (auto &&x : p.get_parameter_shapes())
    //    {
    //        std::cout << "name is = " << x.first << std::endl;
    //        m[x.first] = migraphx::gpu::to_gpu(migraphx::generate_argument(x.second, get_hash(x.first)));
    //    }
 
    //    auto result = migraphx::gpu::from_gpu(p.eval(m));
    //    std::vector<float> resData(4 * 5);
    //    result.visit([&](auto output) { resData.assign(output.begin(), output.end()); });

    //    std::cout << "res = " << std::endl;
    //    for_each(resData.begin(), resData.end(), [](float & i) { std::cout << i << "\t"; });
    //    std::cout << std::endl;
    //}

    return 0;
}
