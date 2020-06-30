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

template <class T>
auto get_hash(const T& x)
{
    return std::hash<T>{}(x);
}

template<typename T>
void print_res(std::vector<T> &res) {
    std::cout << "output size = " << res.size() << std::endl;
    for (size_t i = 0; i < res.size(); i++) {
        std::cout << std::setw(12) << res.at(i) <<",";
        if ((i + 1) % 5 == 0) {
            std::cout << std::endl;
        }
    }
    std::cout << std::endl;
}


void run_cpu(migraphx::program &p, std::vector<float> &res)
{
    p.compile(migraphx::cpu::target{});
    migraphx::program::parameter_map m;
    for (auto&& x : p.get_parameter_shapes())
    {
        std::cout << "x.first = " << x.first << std::endl;
        auto&& argu = migraphx::generate_argument(x.second, get_hash(x.first));
        m[x.first] = argu;
    }

    auto result = p.eval({m});
    result.visit([&](auto output) { res.assign(output.begin(), output.end()); });

    std::cout << "cpu res = " << std::endl;
    print_res(res);
    std::cout << std::endl;
}

void run_gpu(migraphx::program &p, std::vector<float> &res)
{
    p.compile(migraphx::gpu::target{});
    migraphx::program::parameter_map m;
    for (auto && x : p.get_parameter_shapes())
    {
        std::cout << "gpu, x.first = " << x.first << std::endl;
        auto&& argu = migraphx::generate_argument(x.second, get_hash(x.first));
        m[x.first] = migraphx::gpu::to_gpu(argu);
    }
    auto result = migraphx::gpu::from_gpu(p.eval({m}));
    result.visit([&](auto output) { res.assign(output.begin(), output.end()); });
    std::cout << "gpu res = " << std::endl;
    print_res(res);
    std::cout << std::endl;
}

migraphx::program create_program()
{
    migraphx::shape s{migraphx::shape::float_type, {2, 3, 4, 5}};
    std::vector<float> data(s.elements());
    std::iota(data.begin(), data.end(), 1.0);
    std::vector<int64_t> perm{0, 2, 1, 3};

    migraphx::program p;
    //auto lit = p.add_literal(migraphx::literal{s, data});
    auto lit = p.add_parameter("seq", s);
    auto tlit = p.add_instruction(migraphx::op::transpose{perm}, lit);
    std::vector<int64_t> out_shape{0, 0, -1};
    p.add_instruction(migraphx::op::reshape{out_shape}, tlit);

    return p;
}

int main(int argc, char **argv) {

    std::vector<float> cpu_res, gpu_res;
    auto p1 = create_program();
    run_cpu(p1, cpu_res);
    auto p2 = create_program();
    run_gpu(p2, gpu_res);

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


