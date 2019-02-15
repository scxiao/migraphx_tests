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

template<typename T>
void print_res(std::vector<T> &res, std::size_t nd, std::size_t batch_size, std::size_t hidden_size) {
    for (size_t i = 0; i < res.size(); i++) {
        std::cout << std::setw(12) << res.at(i) <<",";

        if (hidden_size > 5) {
            if ((i + 1) % 5 == 0) {
                std::cout << std::endl;
            }
        }
        if ((i + 1) % hidden_size == 0) {
            std::cout << std::endl;
        }


        if (nd == 2) {
            if ((i + 1) % (batch_size * hidden_size * nd) == 0) {
                std::cout << std::endl;
            }
        }
        if ((i + 1) % (batch_size * hidden_size) == 0) {
            std::cout << std::endl;
        }
    }
}


void run_cpu(migraphx::program &p, std::vector<float> &res)
{
    p.compile(migraphx::cpu::target{});

    migraphx::program::parameter_map m;
    std::vector<std::vector<float>> data(p.get_parameter_shapes().size());
    int index = 0;
    std::size_t batch_size, hidden_size, nd;
    batch_size = hidden_size = nd = 1;
    for (auto &&x : p.get_parameter_shapes())
    {
        std::cout << x.first << "'s shape = " << x.second << std::endl;
        if (x.first == std::string("input")) {
            data[index].resize(x.second.elements(), 0.0);
            data[index][0] = data[index][1] = 1.0f;
        }
        else {
            data[index].resize(x.second.elements(), 1.0);
            if (x.first == "1") {
                auto lens = x.second.lens();
                batch_size = lens[1];
                hidden_size = lens[2];
                nd = lens[0];
            }
        }
        auto &&argu = migraphx::argument(x.second, data[index++].data());
        m[x.first] = argu;
    }

    std::cout << "nd = " << nd << ", batch_size = " << batch_size << ", hidden_size = " << hidden_size << std::endl;
    if (batch_size == 0) {
        batch_size = 3;
    }
    if (hidden_size == 0)
    {
        hidden_size = 5;
    }

    auto result = p.eval(m);
    result.visit([&](auto output) { res.assign(output.begin(), output.end()); });

    std::cout << "cpu res = " << std::endl;
    print_res(res, nd, batch_size, hidden_size);
    std::cout << std::endl;
}

void run_gpu(migraphx::program &p, std::vector<float> &res)
{
    p.compile(migraphx::gpu::target{});

    migraphx::program::parameter_map m;
    std::vector<std::vector<float>> data(p.get_parameter_shapes().size());
    int index = 0;
    std::size_t batch_size, hidden_size, nd;
    batch_size = hidden_size = nd = 1;
    for (auto &&x : p.get_parameter_shapes())
    {
        std::cout << x.first << "'s shape = " << x.second << std::endl;
        if (x.first == std::string("input")) {
            data[index].resize(x.second.elements(), 0.0);
            data[index][0] = data[index][1] = 1.0f;
        }
        else {
            data[index].resize(x.second.elements(), 1.0f);
            if (x.first == "1") {
                auto lens = x.second.lens();
                batch_size = lens[1];
                hidden_size = lens[2];
                nd = lens[0];
            }
        }

        auto&& argu = migraphx::argument(x.second, data[index++].data());
        m[x.first] = migraphx::gpu::to_gpu(argu);
    }
    auto result = migraphx::gpu::from_gpu(p.eval(m));
    result.visit([&](auto output) { res.assign(output.begin(), output.end()); });

    std::cout << "nd = " << nd << ", batch_size = " << batch_size << ", hidden_size = " << hidden_size << std::endl;
    if (batch_size == 0) {
        batch_size = 3;
    }
    if (hidden_size == 0)
    {
        hidden_size = 5;
    }

    std::cout << "gpu res = " << std::endl;
    print_res(res, nd, batch_size, hidden_size);
    std::cout << std::endl;
}


int main(int argc, char **argv) {
    if (argc < 2) {
        std::cout << "Usage: " << argv[0] << " onnx_file gpu/cpu" << std::endl;
        return 0;
    }

    std::vector<float> cpu_res, gpu_res;
    migraphx::program prog2 = load_onnx_file(argv[1]);
    run_gpu(prog2, gpu_res);
    migraphx::program prog1 = load_onnx_file(argv[1]);
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


