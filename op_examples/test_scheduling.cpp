#include <iostream>
#include <vector>
#include <string>
#include <migraphx/literal.hpp>
#include <migraphx/quantization.hpp>
#include <migraphx/operators.hpp>
#include <migraphx/program.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/gpu/target.hpp>
#include <migraphx/cpu/target.hpp>
#include <migraphx/onnx.hpp>
#include "utilities.hpp"


migraphx::program test_program1()
{
    migraphx::program p;
    migraphx::shape s{migraphx::shape::float_type, {2, 4}};

    auto pa1 = p.add_parameter("a", s);
    auto pb1 = p.add_parameter("b", s);

    auto sum = p.add_instruction(migraphx::op::add{}, pa1, pb1);
    p.add_instruction(migraphx::op::abs{}, sum);
   
    return p;
}

migraphx::program param_prog()
{
    migraphx::program p;
    migraphx::shape s{migraphx::shape::float_type, {2, 4}};

    auto pa1 = p.add_parameter("a", s);
    auto pb1 = p.add_parameter("b", s);

    p.add_return({"x", "y"}, {pa1, pb1});
   
    return p;
}

migraphx::program test_program2()
{
    migraphx::program p;
    migraphx::shape sa1{migraphx::shape::float_type, {2, 4}};
    migraphx::shape sb1{migraphx::shape::float_type, {4, 8}};
    migraphx::shape sa2{migraphx::shape::float_type, {2, 7}};
    migraphx::shape sb2{migraphx::shape::float_type, {7, 8}};
    migraphx::shape sa3{migraphx::shape::float_type, {2, 20}};
    migraphx::shape sb3{migraphx::shape::float_type, {20, 8}};

    auto pa1 = p.add_parameter("a", sa1);
    auto pb1 = p.add_parameter("b", sb1);
    auto pa2 = p.add_parameter("c", sa2);
    auto pb2 = p.add_parameter("d", sb2);
    auto pa3 = p.add_parameter("e", sa3);
    auto pb3 = p.add_parameter("f", sb3);
    auto dot1 = p.add_instruction(migraphx::op::dot{}, pa1, pb1);
    auto dot2 = p.add_instruction(migraphx::op::dot{}, pa2, pb2);
    auto dot3 = p.add_instruction(migraphx::op::dot{}, pa3, pb3);
    auto sum1 = p.add_instruction(migraphx::op::mul{}, dot1, dot2);
    auto sum2 = p.add_instruction(migraphx::op::mul{}, dot2, dot3);
    p.add_instruction(migraphx::op::add{}, sum1, sum2);
    
    return p;
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

migraphx::program rnn_bi()
{
    std::size_t batch_size  = 2;
    std::size_t seq_len     = 10;
    std::size_t hidden_size = 4;
    std::size_t input_size  = 3;
    std::size_t num_dirct   = 2;
    float clip              = 0.0f;

    migraphx::program p;
    migraphx::shape in_shape{migraphx::shape::float_type, {seq_len, batch_size, input_size}};
    migraphx::shape w_shape{migraphx::shape::float_type, {num_dirct, hidden_size, input_size}};
    migraphx::shape r_shape{migraphx::shape::float_type, {num_dirct, hidden_size, hidden_size}};
    migraphx::shape b_shape{migraphx::shape::float_type, {num_dirct, 2 * hidden_size}};
    migraphx::shape ih_shape{migraphx::shape::float_type, {num_dirct, batch_size, hidden_size}};

    auto seq = p.add_parameter("seq", in_shape);
    auto w   = p.add_parameter("w", w_shape);
    auto r   = p.add_parameter("r", r_shape);
    auto hs =
        p.add_instruction(migraphx::op::rnn{hidden_size,
                                            {migraphx::op::tanh{}, migraphx::op::tanh{}},
                                            migraphx::op::rnn_direction::bidirectional,
                                            clip},
                          seq,
                          w,
                          r);
    auto last_hs = p.add_instruction(migraphx::op::rnn_last_output{}, hs);
    p.add_return({}, {hs, last_hs});

    return p;
}

void load_onnx_file(std::string file_name) {
    auto prog = migraphx::parse_onnx(file_name);
    std::cout << "Load program is: " << std::endl;
    std::cout << prog << std::endl;
    //migraphx::capture_arguments(prog, migraphx::cpu::target{});
    //std::vector<std::string> op_names = {"convolution", "dot"};
    //std::vector<std::pair<float, float>> quant_params(300, std::make_pair<float, float>(1.0f, 0.0f));
    //migraphx::quantize_int8(prog, op_names, quant_params);
    //std::cout << "Quantized program is: " << std::endl;
    //std::cout << prog << std::endl;
    prog.compile(migraphx::gpu::target{});
}

migraphx::program trans_tanh1()
{
    migraphx::program p;
    auto x  = p.add_parameter("x", migraphx::shape{migraphx::shape::float_type, {4, 3, 3, 3}});
    auto tx = p.add_instruction(migraphx::op::transpose{{0, 1, 3, 2}}, x);
    auto tanhx = p.add_instruction(migraphx::op::tanh{}, tx);
    auto r     = p.add_instruction(migraphx::op::add{}, tanhx, tanhx);
    p.add_return({}, {tx, r});

    return p;
}


int main(int argc, char **argv) {
    //if (argc != 2) {
    //    std::cout << "Usage: " << argv[0] << " onnx_file" << std::endl;
    //    return 0;
    //}

    //load_onnx_file(argv[1]);
    auto p = trans_tanh1();

    std::cout << "prog = " << std::endl;
    std::cout << p << std::endl;

    //p.compile(migraphx::gpu::target{});

    std::vector<float> cpu_res, gpu_res;
    run_prog(p, migraphx::cpu::target{}, cpu_res);
    run_prog(p, migraphx::gpu::target{}, gpu_res);
    //run_gpu(prog, gpu_res);
    bool ret2 = compare_results(cpu_res, gpu_res);
    std::cout << (ret2 ? "PASSED!" : "FAILED") << std::endl;

    return 0;
}

