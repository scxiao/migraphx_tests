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
    migraphx::program p;

    std::vector<float> data(4 * 5);
    std::iota(data.begin(), data.end(), 0);
    migraphx::shape s{migraphx::shape::double_type, {4, 5}};

    auto a0 = p.add_literal(migraphx::literal{s, data});
    p.add_instruction(migraphx::op::sin{}, a0);
    p.compile(migraphx::gpu::target{});
    migraphx::program::parameter_map m;
    for (auto &&x : p.get_parameter_shapes())
    {
        m[x.first] = migraphx::gpu::to_gpu(migraphx::generate_argument(x.second, get_hash(x.first)));
    }
    auto result = migraphx::gpu::from_gpu(p.eval(m));
    std::vector<float> resData(4 * 5);
    result.visit([&](auto output) { resData.assign(output.begin(), output.end()); });

    std::cout << "res = " << std::endl;
    for_each(resData.begin(), resData.end(), [](float& i) { std::cout << i << "\t"; });
    std::cout << std::endl;

    return 0;
}
