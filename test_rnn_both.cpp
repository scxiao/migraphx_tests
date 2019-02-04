#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <migraphx/literal.hpp>
#include <migraphx/operators.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/program.hpp>
#include <migraphx/cpu/target.hpp>
#include <migraphx/gpu/target.hpp>
#include <migraphx/gpu/hip.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/onnx.hpp>

migraphx::program load_onnx_file(std::string file_name) {
    auto prog = migraphx::parse_onnx(file_name);

    std::cout << "prog = " << std::endl;
    std::cout << prog << std::endl;

    return prog;
}

void run_cpu(migraphx::program &p, std::vector<float> &res)
{
    p.compile(migraphx::cpu::target{});

    migraphx::program::parameter_map m;
    std::vector<float> data;
    for (auto &&x : p.get_parameter_shapes())
    {
        data.resize(x.second.elements(), 0.0);
        if (x.first == std::string("input")) {
            data[0] = data[1] = 1.0f;
        }
        auto &&argu = migraphx::argument(x.second, data.data());
        m[x.first] = argu;
    }

    auto result = p.eval(m);
    result.visit([&](auto output) { res.assign(output.begin(), output.end()); });

    std::cout << "cpu res = " << std::endl;
    for (size_t i = 0; i < res.size(); i++) {
        std::cout << std::setw(12) << res.at(i) <<",";
        if ((i + 1) % 5 == 0) {
            std::cout << std::endl;
        }
    }
    std::cout << std::endl;
}

void run_gpu(migraphx::program &p, std::vector<float> &res)
{
    p.compile(migraphx::gpu::target{});

    migraphx::program::parameter_map m;
    std::vector<float> data;
    for (auto &&x : p.get_parameter_shapes())
    {
        data.resize(x.second.elements(), 0.0);
        if (x.first == std::string("input")) {
            data[0] = data[1] = 1.0f;
        }
        auto&& argu = migraphx::argument(x.second, data.data());
        m[x.first] = migraphx::gpu::to_gpu(argu);
    }
    auto result = migraphx::gpu::from_gpu(p.eval(m));
    result.visit([&](auto output) { res.assign(output.begin(), output.end()); });

    std::cout << "gpu res = " << std::endl;
    for (size_t i = 0; i < res.size(); i++) {
        std::cout << std::setw(12) << res.at(i) <<",";
        if ((i + 1) % 5 == 0) {
            std::cout << std::endl;
        }
    }
    std::cout << std::endl;
}


int main(int argc, char **argv) {
    if (argc < 2) {
        std::cout << "Usage: " << argv[0] << " onnx_file gpu/cpu" << std::endl;
        return 0;
    }

    std::vector<float> cpu_res, gpu_res;
    migraphx::program prog1 = load_onnx_file(argv[1]);
    run_cpu(prog1, cpu_res);
    migraphx::program prog2 = load_onnx_file(argv[1]);
    run_gpu(prog2, gpu_res);

    std::size_t cpu_size = cpu_res.size();
    std::size_t gpu_size = gpu_res.size();
    if (cpu_size != gpu_size) {
        std::cout << "output size mistach!!!!!!!!!!!!!!!!" << std::endl;
    }

    for (std::size_t i = 0; i < cpu_size; i++) {
        if (fabs(cpu_res[i] - gpu_res[i]) > 1.0e-6)
        {
            std::cout << "cpu_result[" << i << "] (" << cpu_res[i] << ") != gpu_result[" << i << "] (" <<
                gpu_res[i] << ")!!!!!!" << std::endl;
        }
    }
    std::cout << "PASSED!!!" << std::endl;

    return 0;
}


