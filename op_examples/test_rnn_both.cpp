#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <migraphx/literal.hpp>
#include <migraphx/operators.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/program.hpp>
#include <migraphx/gpu/hip.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/onnx.hpp>
#include "utilities.hpp"

migraphx::program create_program()
{
    std::size_t batch_size  = 2;
    std::size_t seq_len     = 3;
    std::size_t hidden_size = 5;
    std::size_t input_size  = 8;
    std::size_t num_dirct   = 1;
    float clip              = 0.0f;

    migraphx::program p;
    migraphx::shape in_shape{migraphx::shape::float_type, {seq_len, batch_size, input_size}};
    migraphx::shape w_shape{migraphx::shape::float_type,
                            {num_dirct, 3 * hidden_size, input_size}};
    migraphx::shape r_shape{migraphx::shape::float_type,
                            {num_dirct, 3 * hidden_size, hidden_size}};
    migraphx::shape b_shape{migraphx::shape::float_type, {num_dirct, 6 * hidden_size}};
    migraphx::shape ih_shape{migraphx::shape::float_type, {num_dirct, batch_size, hidden_size}};

    auto seq  = p.add_parameter("seq", in_shape);
    auto w    = p.add_parameter("w", w_shape);
    auto r    = p.add_parameter("r", r_shape);
    auto bias = p.add_parameter("bias", b_shape);
    auto ih   = p.add_parameter("ih", ih_shape);
    auto und  = p.add_instruction(migraphx::op::undefined{});

    auto output =
        p.add_instruction(migraphx::op::gru{hidden_size,
                                            {migraphx::op::sigmoid{}, migraphx::op::tanh{}},
                                            migraphx::op::rnn_direction::forward,
                                            clip},
                          seq,
                          w,
                          r,
                          bias,
                          und,
                          ih);
    p.add_instruction(migraphx::op::rnn_last_output{}, output);

    return p;
}

migraphx::program load_onnx_file(std::string file_name) {
    auto prog = migraphx::parse_onnx(file_name);

    std::cout << "prog = " << std::endl;
    std::cout << prog << std::endl;

    return prog;
}

migraphx::program rnn_forward10()
{
    std::size_t batch_size  = 2;
    std::size_t seq_len     = 10;
    std::size_t hidden_size = 4;
    std::size_t input_size  = 3;
    std::size_t num_dirct   = 1;
    float clip              = 0.0f;

    migraphx::program p;
    migraphx::shape in_shape{migraphx::shape::float_type, {seq_len, batch_size, input_size}};
    migraphx::shape w_shape{migraphx::shape::float_type, {num_dirct, hidden_size, input_size}};
    migraphx::shape r_shape{migraphx::shape::float_type, {num_dirct, hidden_size, hidden_size}};
    migraphx::shape b_shape{migraphx::shape::float_type, {num_dirct, 2 * hidden_size}};
    migraphx::shape ih_shape{migraphx::shape::float_type, {num_dirct, batch_size, hidden_size}};

    auto seq  = p.add_parameter("seq", in_shape);
    auto w    = p.add_parameter("w", w_shape);
    auto r    = p.add_parameter("r", r_shape);
    auto bias = p.add_parameter("bias", b_shape);
    auto ih   = p.add_parameter("ih", ih_shape);
    auto und  = p.add_instruction(migraphx::op::undefined{});

    auto output =
        p.add_instruction(migraphx::op::rnn{hidden_size,
                                            {migraphx::op::tanh{}, migraphx::op::tanh{}},
                                            migraphx::op::rnn_direction::forward,
                                            clip},
                          seq,
                          w,
                          r,
                          bias,
                          und,
                          ih);
    p.add_instruction(migraphx::op::rnn_last_output{}, output);

    return p;
}

int main(int argc, char **argv) {

    std::vector<float> cpu_res, gpu_res;
	auto prog = rnn_forward10();
	
    run_cpu(prog, cpu_res);
    run_gpu(prog, gpu_res);

    bool ret2 = compare_results(cpu_res, gpu_res);
    std::cout << (ret2 ? "PASSED!" : "FAILED") << std::endl;

    return 0;
}


