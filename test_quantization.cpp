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

migraphx::program create_program()
{
    migraphx::program p;
    std::vector<float> a = {0.7481789 ,  0.02906279,  1.01193836,  1.60222907,  1.89135978,
        0.30054158, -0.4892588 , -0.27027533};
    std::vector<float> b = {-0.25829116,  0.27908929, -1.27888957,  0.21152361,  0.08593658,
        0.52163899,  1.38343824, -0.2342857};
    migraphx::shape a_shape{migraphx::shape::float_type, {8}};
    auto al = p.add_literal(migraphx::literal{a_shape, a});
    migraphx::shape b_shape{migraphx::shape::float_type, {8}};
    auto bl = p.add_literal(migraphx::literal{b_shape, b});
    auto sl = p.add_instruction(migraphx::op::add{}, al, bl);
    auto s1l = p.add_instruction(migraphx::op::add{}, sl, al);
    auto s2l = p.add_instruction(migraphx::op::add{}, sl, bl);
    p.add_instruction(migraphx::op::add{}, s1l, s2l);


    return p;
}

migraphx::program create_program1()
{
    migraphx::program p;
    std::vector<float> a = {0.7481789 ,  0.02906279,  1.01193836,  1.60222907,  1.89135978,
        0.30054158, -0.4892588 , -0.27027533};
    std::vector<float> b = {-0.25829116,  0.27908929, -1.27888957,  0.21152361,  0.08593658,
        0.52163899,  1.38343824, -0.2342857};
    migraphx::shape a_shape{migraphx::shape::float_type, {8}};
    auto al = p.add_literal(migraphx::literal{a_shape, a});
    migraphx::shape b_shape{migraphx::shape::float_type, {8}};
    auto bl = p.add_parameter("b", b_shape);
    auto sl = p.add_instruction(migraphx::op::add{}, al, bl);
    auto s1l = p.add_instruction(migraphx::op::add{}, sl, al);
    auto s2l = p.add_instruction(migraphx::op::add{}, sl, bl);
    p.add_instruction(migraphx::op::add{}, s1l, s2l);


    return p;
}

migraphx::program create_program2()
{
    migraphx::program p;
    std::vector<float> a = {0.7481789 ,  0.02906279,  1.01193836,  1.60222907,  1.89135978,
        0.30054158, -0.4892588 , -0.27027533};
    migraphx::shape a_shape{migraphx::shape::float_type, {8}};
    auto al = p.add_literal(migraphx::literal{a_shape, a});
    auto sl = p.add_instruction(migraphx::op::add{}, al, al);
    auto s1l = p.add_instruction(migraphx::op::add{}, sl, sl);


    return p;
}

migraphx::program create_program3()
{
    migraphx::program p;
    std::vector<double> a = {0.7481789 ,  0.02906279,  1.01193836,  1.60222907,  1.89135978,
        0.30054158, -0.4892588 , -0.27027533};
    std::vector<double> b = {-0.25829116,  0.27908929, -1.27888957,  0.21152361,  0.08593658,
        0.52163899,  1.38343824, -0.2342857};
    migraphx::shape a_shape{migraphx::shape::double_type, {8}};
    auto al = p.add_literal(migraphx::literal{a_shape, a});
    migraphx::shape b_shape{migraphx::shape::double_type, {8}};
    auto bl = p.add_literal(migraphx::literal{b_shape, b});
    auto sl = p.add_instruction(migraphx::op::add{}, al, bl);
    auto s1l = p.add_instruction(migraphx::op::add{}, sl, al);
    auto s2l = p.add_instruction(migraphx::op::add{}, sl, bl);
    p.add_instruction(migraphx::op::add{}, s1l, s2l);


    return p;
}

migraphx::program create_program_mm_c33(float alpha = 1.0, float beta = 1.0f)
{
    migraphx::program p;
    std::vector<float> a = {-0.86217194, -1.04129542, -0.64850364, -0.97078327,
       -0.40516386,  0.83136927,  0.37717502,  0.42271939,
        1.10062165, -0.92239359,  0.40403076, -0.43935377};
    std::vector<float> b = {0.76084386,  1.89201125,  1.73218067,
        0.7148568 , -0.55578914,  0.05799101,
       -1.24090721, -0.51151978,  1.13255803,
        0.21540723, -1.10459009,  0.45580331};
	std::vector<float> c = {-0.80473623,  0.35154171, -2.73077756,
       -0.09093885, -1.88850472, -0.03375556,
       -0.41798276,  2.87368099,  2.11031439};

    migraphx::shape a_shape{migraphx::shape::float_type, {3, 4}};
    auto al = p.add_literal(migraphx::literal{a_shape, a});
    migraphx::shape b_shape{migraphx::shape::float_type, {4, 3}};
    auto bl = p.add_literal(migraphx::literal{b_shape, b});
    migraphx::shape c_shape{migraphx::shape::float_type, {3, 3}};
    std::vector<float> vec_c(3 * 3, 4.0f);
    auto cl = p.add_literal(migraphx::literal{c_shape, c});
    p.add_instruction(migraphx::op::dot{alpha, beta}, al, bl, cl);

    return p;
}

migraphx::argument copy(const migraphx::argument& arg)
{
    return arg;
}

auto create_reduce_mean()
{
    migraphx::program p;
    auto input =
        p.add_parameter("x", migraphx::shape{migraphx::shape::double_type, {4, 3, 3, 3}});
    p.add_instruction(migraphx::op::reduce_sum{{1}}, input);

    return p;
}


void dot_large_alpha_beta_float()
{
    auto create_program = [] {
        migraphx::program p;
        migraphx::shape sa{migraphx::shape::float_type, {2, 16}};
        migraphx::shape sb{migraphx::shape::float_type, {16, 8}};
        migraphx::shape sc{migraphx::shape::float_type, {2, 8}};
        auto pa = p.add_parameter("a", sa);
        auto pb = p.add_parameter("b", sb);
        auto pc = p.add_parameter("c", sc);

        p.add_instruction(migraphx::op::dot{20.0f, 50.5f}, pa, pb, pc);

        return p;
    };

    auto p = create_program();

    const std::vector<std::pair<float, float>>& quant_params{{0.1f, 1.0f}, {0.1f, 0.0f}, {0.1f, 100.0f}};
    // default scale 64.0f is used for all args
    migraphx::quantize_int8(p, quant_params, {"dot"});

    std::cout << "quantized program = " << std::endl;
    std::cout << p << std::endl;
}

void dot_float()
{
    auto create_program = [] {
        migraphx::program p;
        migraphx::shape sa{migraphx::shape::float_type, {2, 16}};
        migraphx::shape sb{migraphx::shape::float_type, {16, 8}};
        migraphx::shape sc{migraphx::shape::float_type, {2, 8}};
        auto pa = p.add_parameter("a", sa);
        auto pb = p.add_parameter("b", sb);
        auto pc = p.add_parameter("c", sc);

        p.add_instruction(migraphx::op::dot{2.0f, 1.5f}, pa, pb, pc);

        return p;
    };


    auto p = create_program();
    const std::vector<std::pair<float, float>>& quant_params{
        {0.1f, 0.0f}, {0.1f, 0.0f}, {0.1f, 100.0f}};
    // default scale 64.0f is used for all args
    migraphx::quantize_int8(p, quant_params, {"dot"});

    std::cout << "quantized program = " << std::endl;
    std::cout << p << std::endl;
}

void dot_int32()
{
    auto create_program = [] {
        migraphx::program p;
        migraphx::shape sa{migraphx::shape::int32_type, {2, 16}};
        migraphx::shape sb{migraphx::shape::int32_type, {16, 8}};
        migraphx::shape sc{migraphx::shape::int32_type, {2, 8}};
        auto pa = p.add_parameter("a", sa);
        auto pb = p.add_parameter("b", sb);
        auto pc = p.add_parameter("c", sc);

        p.add_instruction(migraphx::op::dot{2.0f, 5.5f}, pa, pb, pc);

        return p;
    };

    auto p = create_program();

    const std::vector<std::pair<float, float>>& quant_params{{0.1f, 1.0f}, {0.1f, 0.0f}, {0.1f, 100.0f}};
    // default scale 64.0f is used for all args
    migraphx::quantize_int8(p, quant_params, {"dot"});

    std::cout << "quantized program = " << std::endl;
    std::cout << p << std::endl;
}

void dot_double()
{
    auto create_program = [] {
        migraphx::program p;
        migraphx::shape sa{migraphx::shape::double_type, {2, 16}};
        migraphx::shape sb{migraphx::shape::double_type, {16, 8}};
        migraphx::shape sc{migraphx::shape::double_type, {2, 8}};
        auto pa = p.add_parameter("a", sa);
        auto pb = p.add_parameter("b", sb);
        auto pc = p.add_parameter("c", sc);

        p.add_instruction(migraphx::op::dot{2.0f, 5.5f}, pa, pb, pc);

        return p;
    };

    auto p = create_program();

    const std::vector<std::pair<float, float>>& quant_params{{0.1f, 1.0f}, {0.1f, 0.0f}, {0.1f, 100.0f}};
    // default scale 64.0f is used for all args
    migraphx::quantize_int8(p, quant_params, {"dot"});

    std::cout << "quantized program = " << std::endl;
    std::cout << p << std::endl;
}

void dot_double_2args()
{
    auto create_program = [] {
        migraphx::program p;
        migraphx::shape sa{migraphx::shape::double_type, {2, 16}};
        migraphx::shape sb{migraphx::shape::double_type, {16, 8}};
        auto pa = p.add_parameter("a", sa);
        auto pb = p.add_parameter("b", sb);

        p.add_instruction(migraphx::op::dot{2.0f, 1.5f}, pa, pb);

        return p;
    };

    auto p = create_program();
    const std::vector<std::pair<float, float>>& quant_params{
        {0.1f, 0.0f}, {0.1f, 0.0f}};
    migraphx::quantize_int8(p, quant_params, {"dot"});

    std::cout << "quantized program = " << std::endl;
    std::cout << p << std::endl;
}

void conv_float()
{
    auto create_program = []
    {
        migraphx::program p;
        auto input =
            p.add_parameter("x", migraphx::shape{migraphx::shape::float_type, {4, 3, 3, 3}});
        auto weights =
            p.add_parameter("w", migraphx::shape{migraphx::shape::float_type, {4, 3, 3, 3}});
        p.add_instruction(migraphx::op::convolution{}, input, weights);

        return p;
    };

    auto p = create_program();
    const std::vector<std::pair<float, float>>& quant_params{{0.1f, 0.0f}, {0.1f, 0.0f}};
    migraphx::quantize_int8(p, quant_params, {"convolution"});

    std::cout << "quantized program = " << std::endl;
    std::cout << p << std::endl;
}

void conv_double()
{
    auto create_program = []
    {
        migraphx::program p;
        auto input =
            p.add_parameter("x", migraphx::shape{migraphx::shape::double_type, {4, 3, 3, 3}});
        auto weights =
            p.add_parameter("w", migraphx::shape{migraphx::shape::double_type, {4, 3, 3, 3}});
        p.add_instruction(migraphx::op::convolution{}, input, weights);

        return p;
    };

    auto p = create_program();
    const std::vector<std::pair<float, float>>& quant_params{{0.1f, 0.0f}, {0.1f, 0.0f}};
    migraphx::quantize_int8(p, quant_params, {"convolution"});

    std::cout << "quantized program = " << std::endl;
    std::cout << p << std::endl;
}

void dot_int32_one_arg()
{
    auto create_program = [] {
        migraphx::program p;
        migraphx::shape s{migraphx::shape::int32_type, {16, 16}};
        auto pa = p.add_parameter("a", s);

        p.add_instruction(migraphx::op::dot{20.0f, 50.0f}, pa, pa);

        return p;
    };

    auto p = create_program();
    const std::vector<std::pair<float, float>>& quant_params{
        {1.0f, 1.0f}};
    migraphx::quantize_int8(p, quant_params, {"dot"});
    
    std::cout << "prog = " << std::endl;
    std::cout << p << std::endl;
}

void dot_int32_convert()
{
    auto create_program = [] {
        migraphx::program p;
        migraphx::shape sa{migraphx::shape::int8_type, {2, 16}};
        migraphx::shape sb{migraphx::shape::float_type, {16, 8}};
        auto pa = p.add_parameter("a", sa);
        auto pb = p.add_parameter("b", sb);

        auto fpa = p.add_instruction(migraphx::op::convert{migraphx::shape::float_type}, pa);
        p.add_instruction(migraphx::op::dot{2.0f, 5.5f}, fpa, pb);

        return p;
    };

    auto p = create_program();
    const std::vector<std::pair<float, float>>& quant_params{
        {0.1f, 1.0f}, {0.1f, 0.0f}};
    migraphx::quantize_int8(p, quant_params, {"dot"});

    std::cout << "prog = " << std::endl;
    std::cout << p << std::endl;
}

int main(int argc, char **argv) {
//    std::vector<float> cpu_res, gpu_res;
//    auto p1 = create_reduce_mean();
//    migraphx::target t = migraphx::cpu::target{};
//    //migraphx::capture_arguments(p1, migraphx::cpu::target{});
//    //migraphx::capture_arguments(p1, t);
//    std::cout << "p1 = \n" << p1 << std::endl;
//    //migraphx::quantize(p1, {"dot"});
//    run_cpu(p1, gpu_res);
//
//    //migraphx::shape s{migraphx::shape::float_type, {2, 3}};
//    //migraphx::argument argu = migraphx::generate_argument(s);
//    //std::cout << "argument = " << argu << std::endl;
//    //migraphx::argument argu1 = t.copy_to(argu);
//    //std::cout << "argument1 = " << argu1 << std::endl;
//    //std::cout << "argument = " << argu << std::endl;
//
//    
//
//    auto p2 = create_reduce_mean();
//    migraphx::target gt = migraphx::gpu::target{};
//    //migraphx::capture_arguments(p2, gt);
//    //std::cout << "p2 = \n" << p2 << std::endl;
//    //migraphx::quantize(p2, {"dot"});
//    run_gpu(p2, cpu_res);
//    
//    bool res = compare_results(cpu_res, gpu_res);
//    std::cout << (res ? "PASSED" : "FAILED") << std::endl;
    //dot_large_alpha_beta_float();
	//dot_float();
    //dot_int32();
    //conv_double();
    //dot_int32_one_arg();
	dot_int32_convert();

    return 0;
}

