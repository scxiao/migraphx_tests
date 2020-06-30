#ifndef _TEST_UTILITIES_HPP_
#define _TEST_UTILITIES_HPP_

#include <iostream>
#include <fstream>
#include <iomanip>
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

template<typename T>
void print_res(const T& res)
{
    for (std::size_t i = 0; i < res.size(); ++i)
    {
        std::cout << std::setw(12) << res[i] << ", ";
        if ((i + 1) % 6 == 0) {
            std::cout << std::endl;
        }
    }
}

template <class T>
auto get_hash(const T& x)
{
    return std::hash<T>{}(x);
}

migraphx::argument gen_argument(migraphx::shape s, unsigned long seed)
{
    migraphx::argument result;
    s.visit_type([&](auto as) {
        using type = typename decltype(as)::type;
		std::vector<type> v(s.elements());
        std::srand(seed);
		for_each(v.begin(), v.end(), [&](auto &val) { val = 1.0 * std::rand()/(RAND_MAX); } );
        //std::cout << v[0] << "\t" << v[1] << "\t" << v[2] << std::endl;
        result     = {s, [v]() mutable { return reinterpret_cast<char*>(v.data()); }};
    });

    return result;
}

template<class T>
void run_cpu(migraphx::program p, std::vector<T> &resData)
{
    p.compile(migraphx::cpu::target{});

    migraphx::program::parameter_map m;
    for (auto &&x : p.get_parameter_shapes())
    {
        std::cout << "cpu input: " << x.first << "\'shape = " << x.second << std::endl;
        //auto &&argu = gen_argument(x.second, get_hash(x.first));
        auto &&argu = migraphx::generate_argument(x.second, get_hash(x.first));
        m[x.first] = argu;
        //std::cout << "cpu_arg = " << argu << std::endl;
    }
    auto result = p.eval(m);
    result.visit([&](auto output) { resData.assign(output.begin(), output.end()); });

    std::cout << "cpu output_shape = " << result.get_shape() << std::endl;
    std::cout << "cpu res = " << std::endl;
    print_res(resData);
    std::cout << std::endl;
}

template <class T>
void run_gpu(migraphx::program p, std::vector<T> &resData)
{
    p.compile(migraphx::gpu::target{});

    migraphx::program::parameter_map m;
    for (auto &&x : p.get_parameter_shapes())
    {
        std::cout << "gpu input: " << x.first << "\'shape = " << x.second << std::endl;
        //auto&& argu = gen_argument(x.second, get_hash(x.first));
        auto&& argu = migraphx::generate_argument(x.second, get_hash(x.first));
        //std::cout << "gpu_arg = " << argu << std::endl;
        m[x.first] = migraphx::gpu::to_gpu(argu);
    }

    auto result = migraphx::gpu::from_gpu(p.eval(m));

    result.visit([&](auto output) { resData.assign(output.begin(), output.end()); });


    std::cout << "gpu output_shape = " << result.get_shape() << std::endl;
    std::cout << "gpu res = " << std::endl;
    print_res(resData);
    std::cout << std::endl;
}

template<typename T>
bool compare_results(const T& cpu_res, const T& gpu_res)
{
    bool passed = true;
    std::size_t cpu_size = cpu_res.size();
    float fmax_diff = 0.0f;
    size_t max_index = 0;
    for (std::size_t i = 0; i < cpu_size; i++) {
        auto diff = fabs(cpu_res[i] - gpu_res[i]);
        if (diff > 1.0e-6)
        {
            if (fmax_diff < diff) 
            {
                fmax_diff = diff;
                max_index = i;
                passed = false;
            }
        }
    }

    if (!passed)
    {
        size_t i = max_index;
        std::cout << "cpu_result[" << i << "] (" << cpu_res[i] << ") != gpu_result[" << i << "] (" <<
            gpu_res[i] << ")!!!!!!" << std::endl;

        std::cout << "max_diff = " << fmax_diff << std::endl;
    }

    return passed;
}

bool compare_results(const std::vector<int>&cpu_res, const std::vector<int>& gpu_res)
{
    bool passed = true;
    std::size_t cpu_size = cpu_res.size();
    for (std::size_t i = 0; i < cpu_size; i++) {
        if (cpu_res[i] - gpu_res[i] != 0)
        {
            std::cout << "cpu_result[" << i << "] (" << cpu_res[i] << ") != gpu_result[" << i << "] (" <<
                gpu_res[i] << ")!!!!!!" << std::endl;
            passed = false;
        }
    }

    return passed;
}

#endif

