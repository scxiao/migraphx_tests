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

        std::vector<float> data(4 * 5 * 6);
        std::iota(data.begin(), data.end(), 0);
        migraphx::shape s{migraphx::shape::float_type, {4, 5, 6}};

        auto a0 = p.add_literal(migraphx::literal{s, data});
        auto asi = p.add_instruction(migraphx::op::shape_of{}, a0);
        migraphx::argument arg0 = asi->eval();
        std::cout << "arg0 = " << arg0.get_shape() << std::endl;
        bool ret = is_context_free(asi->get_operator());
        std::cout << "context free: " << ret << std::endl;

        p.compile(migraphx::cpu::target{});
        auto result = p.eval({});
        std::vector<int64_t> resData(4 * 5);
        result.visit([&](auto output) { resData.assign(output.begin(), output.end()); });

        std::cout << "res = " << std::endl;
        for_each(resData.begin(), resData.end(), [](int64_t& i) { std::cout << i << "\t"; });
        std::cout << std::endl;
    }

    {
        migraphx::program p;

        std::vector<float> data(4 * 5 * 6);
        std::iota(data.begin(), data.end(), 0);
        migraphx::shape s{migraphx::shape::float_type, {4, 5, 6}};

        auto a0 = p.add_literal(migraphx::literal{s, data});
        p.add_instruction(migraphx::op::shape_of{}, a0);
        p.compile(migraphx::gpu::target{});

        migraphx::program::parameter_map m;
        for (auto &&x : p.get_parameter_shapes())
        {
            std::cout << "name is = " << x.first << std::endl;
            m[x.first] = migraphx::gpu::to_gpu(migraphx::generate_argument(x.second, get_hash(x.first)));
        }
        auto result = migraphx::gpu::from_gpu(p.eval(m));
        std::vector<int64_t> resData(4 * 5);
        result.visit([&](auto output) { resData.assign(output.begin(), output.end()); });

        std::cout << "res = " << std::endl;
        for_each(resData.begin(), resData.end(), [](int64_t& i) { std::cout << i << "\t"; });
        std::cout << std::endl;
    }


    return 0;
}
