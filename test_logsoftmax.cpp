#include <iostream>
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

template <class T>
auto get_hash(const T& x)
{
    return std::hash<T>{}(x);
}

migraphx::program create_program()
{
    migraphx::program p;

    int axis = 3;
    std::vector<float> data(3 * 4 * 5 * 6);
    std::iota(data.begin(), data.end(), 0.5);
    migraphx::shape s{migraphx::shape::float_type, {3, 4, 5, 6}};
    auto a0 = p.add_literal(migraphx::literal{s, data});
    p.add_instruction(migraphx::op::logsoftmax{axis}, a0);

    return p;
}

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

void run_cpu(migraphx::program &p, std::vector<float> &resData)
{
    p.compile(migraphx::cpu::target{});

    migraphx::program::parameter_map m;
    for (auto &&x : p.get_parameter_shapes())
    {
        auto &&argu = migraphx::generate_argument(x.second, get_hash(x.first));
        m[x.first] = argu;
    }
    auto result = p.eval(m);
    result.visit([&](auto output) { resData.assign(output.begin(), output.end()); });

    std::cout << "cpu output_shape = " << result.get_shape() << std::endl;
    std::cout << "cpu res = " << std::endl;
    print_res(resData);
    std::cout << std::endl;
}

void run_gpu(migraphx::program &p, std::vector<float> &resData)
{
    p.compile(migraphx::gpu::target{});

    migraphx::program::parameter_map m;
    for (auto &&x : p.get_parameter_shapes())
    {
        auto&& argu = migraphx::generate_argument(x.second, get_hash(x.first));
        m[x.first] = migraphx::gpu::to_gpu(argu);
    }
    auto result = migraphx::gpu::from_gpu(p.eval(m));

    result.visit([&](auto output) { resData.assign(output.begin(), output.end()); });


    std::cout << "gpu output_shape = " << result.get_shape() << std::endl;
    std::cout << "gpu res = " << std::endl;
    print_res(resData);
    std::cout << std::endl;
}

int main(int argc, char **argv) {
    std::vector<float> cpu_res, gpu_res;
    migraphx::program prog2 = create_program();
    run_gpu(prog2, gpu_res);
    migraphx::program prog1 = create_program();
    run_cpu(prog1, cpu_res);

    std::size_t cpu_size = cpu_res.size();
    std::size_t gpu_size = gpu_res.size();
    if (cpu_size != gpu_size) {
        std::cout << "output size mistach!!!!!!!!!!!!!!!!" << std::endl;
    }

    bool passed = true;
    for (std::size_t i = 0; i < cpu_size; i++) {
        if (fabs(cpu_res[i] - gpu_res[i]) > 1.0e-6)
        {
            std::cout << "cpu_result[" << i << "] (" << cpu_res[i] << ") != gpu_result[" << i << "] (" <<
                gpu_res[i] << ")!!!!!!" << std::endl;
            passed = false;
        }
    }
    std::cout << (passed ? "PASSED!!!" : "FAILED!!!") << std::endl;

    return 0;
}

