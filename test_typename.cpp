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

    std::vector<float> data(2);
    std::iota(data.begin(), data.end(), 1);
    migraphx::shape s{migraphx::shape::double_type, {2, 1}};

    auto a0 = p.add_literal(migraphx::literal{s, data});
    auto i1 = p.add_instruction(migraphx::op::sin{}, a0);
    auto i2 = p.add_instruction(migraphx::op::cos{}, i1);
    auto i3 = p.add_instruction(migraphx::op::tan{}, i2);
    auto i4 = p.add_instruction(migraphx::op::atan{}, i3);
    auto i5 = p.add_instruction(migraphx::op::acos{}, i4);
    auto i6 = p.add_instruction(migraphx::op::asin{}, i5);
    auto i7 = p.add_instruction(migraphx::op::cosh{}, i6);
    auto i8 = p.add_instruction(migraphx::op::sinh{}, i7);
    auto i9 = p.add_instruction(migraphx::op::exp{}, i8);
    auto i10 = p.add_instruction(migraphx::op::log{}, i2);
    auto i11 = p.add_instruction(migraphx::op::add{}, i10, i9);
    auto i12 = p.add_instruction(migraphx::op::mul{}, i10, i11);
    auto i13 = p.add_instruction(migraphx::op::max{}, i11, i12);
    auto i14 = p.add_instruction(migraphx::op::min{}, i12, i13);
    p.add_instruction(migraphx::op::min{}, i12, i14);
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
