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
#include <migraphx/ref/target.hpp>
#include <migraphx/gpu/target.hpp>
#include <migraphx/gpu/hip.hpp>
#include <migraphx/manage_ptr.hpp>
#include <migraphx/type_name.hpp>
#include "test.hpp"

using parameter_map = migraphx::parameter_map;

template<typename T>
void print_res(const T& res)
{
    for (std::size_t i = 0; i < res.size(); ++i)
    {
        std::cout << std::setprecision(9) << std::setw(12) << res[i] << ", ";
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
		for_each(v.begin(), v.end(), [&](auto val) { val = 1.0 * std::rand()/(RAND_MAX); } );
        //std::cout << v[0] << "\t" << v[1] << "\t" << v[2] << std::endl;
        //result     = {s, [v]() mutable { return reinterpret_cast<char*>(v.data()); }};
    });

    return result;
}

template<class T>
void run_ref(migraphx::program p, std::vector<T> &resData)
{
    p.compile(migraphx::ref::target{});

    parameter_map m;
    for (auto &&x : p.get_parameter_shapes())
    {
        //auto &&argu = gen_argument(x.second, get_hash(x.first));
        auto &&argu = migraphx::generate_argument(x.second, get_hash(x.first));
        m[x.first] = argu;
        //std::cout << "ref_arg = " << argu << std::endl;
    }
    auto result = p.eval(m).back();
    //auto result = p.eval(m);
    result.visit([&](auto output) { resData.assign(output.begin(), output.end()); });

    std::cout << "ref output_shape = " << result.get_shape() << std::endl;
    std::cout << "ref res = " << std::endl;
    print_res(resData);
    std::cout << std::endl;
}

template <class T>
void run_gpu(migraphx::program p, std::vector<T> &resData)
{
    p.compile(migraphx::gpu::target{});

    parameter_map m;
    for (auto &&x : p.get_parameter_shapes())
    {
        std::cout << "gpu input: " << x.first << "\'shape = " << x.second << std::endl;
        //auto&& argu = gen_argument(x.second, get_hash(x.first));
        auto&& argu = migraphx::generate_argument(x.second, get_hash(x.first));
        //std::cout << "gpu_arg = " << argu << std::endl;
        m[x.first] = migraphx::gpu::to_gpu(argu);
    }

    auto result = migraphx::gpu::from_gpu(p.eval(m).back());
    result.visit([&](auto output) { resData.assign(output.begin(), output.end()); });

    //migraphx::gpu::from_gpu(p.eval(m));
    //auto result = migraphx::gpu::from_gpu(m["output"]);
    result.visit([&](auto output) { resData.assign(output.begin(), output.end()); });


    std::cout << "gpu output_shape = " << result.get_shape() << std::endl;
    std::cout << "gpu res = " << std::endl;
    print_res(resData);
    std::cout << std::endl;
}


template<class T>
void print_vec(std::ostream& os, const std::vector<T>& vec, std::size_t column_size)
{
    for (std::size_t i = 0; i < vec.size(); ++i)
    {
        os << vec[i] << "\t";
        if ((i + 1) % column_size == 0)
            os << std::endl;
    }
    os << std::endl;
}

template<class T>
void print_vec(std::vector<T>& vec, std::size_t column_size)
{
    print_vec(std::cout, vec, column_size);
}


template<class T>
std::ostream& operator << (std::ostream& os, const std::vector<T>& vec)
{
    print_vec(os, vec, 8);
}

parameter_map create_param_map(migraphx::program& p)
{
    parameter_map m;
    for (auto&& x : p.get_parameter_shapes())
    {
        if (x.second.type() == migraphx::shape::int32_type or
            x.second.type() == migraphx::shape::int64_type)
        {
            auto&& argu = migraphx::fill_argument(x.second, 0);
            m[x.first] = argu;
        }
        else
        {
            m[x.first] = migraphx::generate_argument(x.second, get_hash(x.first));
        }
    }

    return m;
}

using milliseconds = std::chrono::duration<double, std::milli>;

struct run_options {
    int iter_num = 0;
    migraphx::target t;
    bool offload_copy = false;

    //run_options(int n_iter, const migraphx::target& tg, bool ol_cp) : 
    //    iter_num(n_iter), t(tg), offload_copy(ol_cp) {}
    //run_options(const migraphx::target& tg) : t(tg) { }
};

template <class T>
void run_prog(migraphx::program p, std::vector<std::vector<T>> &resData, const run_options& options)
{
    migraphx::compile_options c_options;
    c_options.offload_copy = options.offload_copy;
    auto& t = options.t;
    p.compile(t, c_options);
    std::cout << "compiled program = " << std::endl;
    std::cout << p << std::endl;
    
    std::string print_name = options.t.name();
    if (print_name == "miopen")
    {
        print_name = "gpu";
    }
    std::cout << "run on " << print_name << "............." << std::endl << std::endl;

    parameter_map m;
    std::vector<int64_t> lens = {16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1};
    std::vector<int> indices = {2, 1, 2, 0, 1, 0};
    for (auto &&x : p.get_parameter_shapes())
    {
        std::cout << "input: " << x.first << "\'shape = " << x.second << std::endl;
        if (x.first == "lengths")
        {
            auto&& argu = migraphx::argument(x.second, lens.data());
            m[x.first] = options.offload_copy ? argu : t.copy_to(argu);
        }
        else if (x.first == "indices")
        {
            auto&& argu = migraphx::argument(x.second, indices.data());
            //std::cout << "argu = " << argu << std::endl;
            m[x.first] = options.offload_copy ? argu : t.copy_to(argu);
        }
        else if (x.second.type() == migraphx::shape::int32_type or
            x.second.type() == migraphx::shape::int64_type)
        {
            auto&& argu = migraphx::fill_argument(x.second, 1);
            std::cout << "argu_int = " << argu << std::endl;
            m[x.first] = options.offload_copy ? argu : t.copy_to(argu);
        }
        else if (x.second.type() == migraphx::shape::bool_type)
        {
            auto&& argu = migraphx::fill_argument(x.second, 1);
            std::cout << "argu_bool = " << argu << std::endl;
            m[x.first] = options.offload_copy ? argu : t.copy_to(argu);
        }
        else
        {
            //auto&& argu = gen_argument(x.second, get_hash(x.first));
            auto&& argu = migraphx::generate_argument(x.second, get_hash(x.first));
            // std::cout << "argu = " << argu << std::endl;
            std::vector<float> vec_arg;
            argu.visit([&](auto v) { vec_arg.assign(v.begin(), v.end()); });
            m[x.first] = options.offload_copy ? argu : t.copy_to(argu);
        }
    }

    if (options.iter_num > 0)
        p.eval(m);

    std::cout << "Begin execution, " << options.iter_num << " iterations...." << std::endl;
    auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < options.iter_num; ++i)
    {
        p.eval(m);
        t.get_context().finish();
    }
    auto end = std::chrono::steady_clock::now();
    auto milli_seconds = std::chrono::duration_cast<milliseconds>(end - start).count();
    std::cout << "time per iteration = " << milli_seconds / options.iter_num << std::endl;
    std::cout << "End execution ...." << std::endl;
    auto results = p.eval(m);

    std::size_t i = 0;
    for (auto&& result : results)
    {
        auto ref_res = t.copy_from(result);
        std::vector<T> resTmp;
        ref_res.visit([&](auto output) { resTmp.assign(output.begin(), output.end()); });
        std::cout << "Output_" << i << "_shape = " << ref_res.get_shape() << std::endl;
        std::cout << "Result_" << i << " = " << std::endl;
        resData.push_back(resTmp);
        print_res(resTmp);
        
        std::cout << std::endl;
        ++i;
    }
}


template<typename T>
bool compare_results(const T& ref_res, const T& gpu_res)
{
    bool passed = true;
    std::size_t ref_size = ref_res.size();
    float fmax_diff = 0.0f;
    size_t max_index = 0;
    for (std::size_t i = 0; i < ref_size; i++) {
        auto diff = fabs(ref_res[i] - gpu_res[i]);
        if (diff > 1.0e-3)
        {
            if (fmax_diff < diff) 
            {
                fmax_diff = diff;
                max_index = i;
                passed = false;
            }
            std::cout << "ref_result[" << i << "] (" << ref_res[i] << ") != gpu_result[" << i << "] (" <<
                gpu_res[i] << ")!!!!!!" << std::endl;
        }
    }

    if (!passed)
    {
        size_t i = max_index;
        std::cout << "ref_result[" << i << "] (" << ref_res[i] << ") != gpu_result[" << i << "] (" <<
            gpu_res[i] << ")!!!!!!" << std::endl;

        std::cout << "max_diff = " << fmax_diff << std::endl;
    }

    return passed;
}

bool compare_results(const std::vector<int>&ref_res, const std::vector<int>& gpu_res)
{
    bool passed = true;
    std::size_t ref_size = ref_res.size();
    for (std::size_t i = 0; i < ref_size; i++) {
        if (ref_res[i] - gpu_res[i] != 0)
        {
            std::cout << "ref_result[" << i << "] (" << ref_res[i] << ") != gpu_result[" << i << "] (" <<
                gpu_res[i] << ")!!!!!!" << std::endl;
            passed = false;
        }
    }

    return passed;
}

bool compare_results(const std::vector<int64_t>&ref_res, const std::vector<int64_t>& gpu_res)
{
    bool passed = true;
    std::size_t ref_size = ref_res.size();
    for (std::size_t i = 0; i < ref_size; i++) {
        if (ref_res[i] - gpu_res[i] != 0)
        {
            std::cout << "ref_result[" << i << "] (" << ref_res[i] << ") != gpu_result[" << i << "] (" <<
                gpu_res[i] << ")!!!!!!" << std::endl;
            passed = false;
        }
    }

    return passed;
}


template<typename T>
std::vector<T> read_input(std::string file_name)
{
    std::vector<T> res;
    std::ifstream ifs(file_name);
    if (!ifs.is_open())
    {
        return {};
    }

    int num;
    while (ifs >> num)
    {
        res.push_back(static_cast<T>(num));
    }
    ifs.close();

    return res;
}


#endif

