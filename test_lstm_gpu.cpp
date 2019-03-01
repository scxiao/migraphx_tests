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

template<typename T>
void print_res(std::vector<T> &res) {
    std::cout << "output size = " << res.size() << std::endl;
    for (size_t i = 0; i < res.size(); i++) {
        std::cout << std::setw(12) << res.at(i) <<",";
        if ((i + 1) % 5 == 0) {
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

    std::cout << "cpu res = " << std::endl;
    print_res(resData);
}

void run_gpu(migraphx::program &p, std::vector<float>& resData)
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

    std::cout << "gpu res = " << std::endl;
    print_res(resData);
}

migraphx::program create_program()
{
    std::size_t batch_size  = 2;
    std::size_t seq_len     = 10;
    std::size_t hidden_size = 4;
    std::size_t input_size  = 3;
    std::size_t num_dirct   = 2;
    float clip = 0.0f;

    migraphx::program p;
    migraphx::shape in_shape{migraphx::shape::float_type, {seq_len, batch_size, input_size}};
    migraphx::shape w_shape{migraphx::shape::float_type, {num_dirct, 4 * hidden_size, input_size}};
    migraphx::shape r_shape{migraphx::shape::float_type, {num_dirct, 4 * hidden_size, hidden_size}};
    migraphx::shape b_shape{migraphx::shape::float_type, {num_dirct, 8 * hidden_size}};
    migraphx::shape ih_shape{migraphx::shape::float_type, {num_dirct, batch_size, hidden_size}};
    std::vector<float> ih_data(ih_shape.elements(), 1.0f);

    auto seq = p.add_parameter("seq", in_shape);
    auto w = p.add_parameter("w", w_shape);
    auto r = p.add_parameter("r", r_shape);
    auto bias = p.add_parameter("bias", b_shape);
    auto ih = p.add_parameter("ih", ih_shape);
    auto ic = p.add_parameter("ic", ih_shape);
    auto und = p.add_instruction(migraphx::op::undefined{});

    auto output = p.add_instruction(migraphx::op::lstm{hidden_size,
                                        {migraphx::op::sigmoid{}, migraphx::op::tanh{}, migraphx::op::tanh{}},
                                        migraphx::op::rnn_direction::bidirectional,
                                        clip},
                      seq,
                      w,
                      r,
                      bias,
                      und,
                      ih,
                      ic,
                      und);

    //p.add_instruction(migraphx::op::lstm_last_cell_output{}, output);
    return p;
}

int main()
{
    std::vector<float> cpu_res, gpu_res;
    auto p2 = create_program();
    run_gpu(p2, gpu_res);
    auto p1 = create_program();
    run_cpu(p1, cpu_res);

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
