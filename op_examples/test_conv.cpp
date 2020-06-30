#include <iostream>
#include <vector>
#include <string>
#include <migraphx/literal.hpp>
#include <migraphx/quantization.hpp>
#include <migraphx/operators.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/program.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/gpu/target.hpp>
#include <migraphx/cpu/target.hpp>
#include <migraphx/onnx.hpp>
#include <iomanip>

template <class T>
auto get_hash(const T& x)
{
    return std::hash<T>{}(x);
}

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

bool compare_results(const std::vector<float>& cpu_res, const std::vector<float>& gpu_res)
{
    bool passed = true;
    std::size_t cpu_size = cpu_res.size();
    float fmax_diff = 0.0f;
    size_t max_index = 0;
    for (std::size_t i = 0; i < cpu_size; i++) {
        auto diff = fabs(cpu_res[i] - gpu_res[i]);
        if (diff > 1.0e-3)
        {
            if (fmax_diff < diff) 
            {
                fmax_diff = diff;
                max_index = i;
                passed = false;
            }
            std::cout << "cpu_result[" << i << "] (" << cpu_res[i] << ") != gpu_result[" << i << "] (" <<
                gpu_res[i] << ")!!!!!!" << std::endl;
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

void run_prog(migraphx::program p, const migraphx::target& t, std::vector<float> &resData)
{
    p.compile(t);
    std::cout << "compiled program = " << std::endl;
    std::cout << p << std::endl;
    std::string print_name = t.name();
    if (print_name == "miopen")
    {
        print_name = "gpu";
    }
    std::cout << "run on " << print_name << "............." << std::endl << std::endl;

    migraphx::program::parameter_map m;
    for (auto &&x : p.get_parameter_shapes())
    {
        std::cout << "input: " << x.first << "\'shape = " << x.second << std::endl;
        auto&& argu = migraphx::generate_argument(x.second, get_hash(x.first));
        m[x.first] = t.copy_to(argu);
    }

    std::cout << "Begin execution ...." << std::endl;
    auto result = t.copy_from(p.eval(m).back());
    std::cout << "End execution ...." << std::endl;

    result.visit([&](auto output) { resData.assign(output.begin(), output.end()); });

    std::cout << "Output_shape = " << result.get_shape() << std::endl;
    print_res(resData);
}

migraphx::program conv_float()
{
    migraphx::program p;
    auto input =
        p.add_parameter("x", migraphx::shape{migraphx::shape::half_type, {1, 3, 20, 20}});
    auto weights =
        p.add_parameter("w", migraphx::shape{migraphx::shape::half_type, {1, 3, 7, 7}});
    p.add_instruction(migraphx::op::convolution{{3, 3}, {1, 1}, {1, 1}}, input, weights);

    return p;
}

int main(int argc, char **argv) {
    auto p = conv_float();

    std::vector<float> cpu_res, gpu_res;
    run_prog(p, migraphx::cpu::target{}, cpu_res);
    run_prog(p, migraphx::gpu::target{}, gpu_res);

    bool b_res = compare_results(cpu_res, gpu_res);

    std::cout << (b_res ? "PASSED!" : "FAILED!") << std::endl;


    return 0;
}

