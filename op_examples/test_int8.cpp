#include <iostream>
#include <vector>
#include <string>
#include <ctime>
#include <cstdlib>
#include <migraphx/literal.hpp>
#include <migraphx/operators.hpp>
#include <migraphx/program.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/gpu/target.hpp>
#include <migraphx/cpu/target.hpp>
#include <migraphx/onnx.hpp>
#include <migraphx/quantization.hpp>
#include "utilities.hpp"

migraphx::program create_conv_int8()
{
    std::size_t n = 2, c = 3, ih = 4, iw = 4;
    migraphx::program p;
    migraphx::shape a_shape{migraphx::shape::int8_type, {n, c, ih, iw}};
    std::vector<int8_t> a(n * c * ih * iw);
    std::iota(a.begin(), a.end(), 0);
    auto al = p.add_literal(migraphx::literal{a_shape, a});
    std::size_t wh = 3, ww = 3;
    std::vector<int8_t> w(n * c * wh * ww);
    std::iota(w.begin(), w.end(), 0);
    migraphx::shape w_shape{migraphx::shape::int8_type, {n, c, wh, ww}};
    auto wl = p.add_literal(migraphx::literal{w_shape, w});

    p.add_instruction(migraphx::op::quant_convolution{{{1, 1}}, {{1, 1}}}, al, wl);
    return p;
}

template<class T>
void fill_array(std::vector<T>& arr)
{
    //std::srand(std::time(nullptr));
    for_each(arr.begin(), arr.end(), [&](auto &val) {
        val = (T)(1 + std::rand()) / (RAND_MAX);
    });
}

migraphx::program create_conv_float()
{
    std::size_t n = 2, c = 3, ih = 4, iw = 4;
    migraphx::program p;
    migraphx::shape a_shape{migraphx::shape::float_type, {n, c, ih, iw}};
    std::vector<float> a(n * c * ih * iw);
    fill_array(a);
    auto al = p.add_literal(migraphx::literal{a_shape, a});
    std::size_t wh = 3, ww = 3;
    std::vector<float> w(n * c * wh * ww);
    fill_array(w);
    migraphx::shape w_shape{migraphx::shape::float_type, {n, c, wh, ww}};
    auto wl = p.add_literal(migraphx::literal{w_shape, w});

    p.add_instruction(migraphx::op::convolution{{{0, 0}}, {{1, 1}}, {{1, 1}}, migraphx::op::valid}, al, wl);

    return p;
}

migraphx::program create_gemm()
{
    migraphx::program p;
    migraphx::shape m1_shape{migraphx::shape::float_type, {3, 4}};
    migraphx::shape m2_shape{migraphx::shape::float_type, {4, 5}};
    migraphx::shape m3_shape{migraphx::shape::float_type, {3, 5}};
    std::vector<float> data1(3 * 4);
    std::vector<float> data2(4 * 5);
    std::vector<float> data3(3 * 5);
    fill_array(data1);
    fill_array(data2);
    fill_array(data3);

    for_each(data1.begin(), data1.end(), [](auto val) {
        std::cout << val << "\t";
    });
    std::cout << std::endl;

    auto l1 = p.add_literal(migraphx::literal{m1_shape, data1});
    auto l2 = p.add_literal(migraphx::literal{m2_shape, data2});
    auto l3 = p.add_literal(migraphx::literal{m3_shape, data3});

    float alpha = 1.0f;
    float beta = 2.0f;

    p.add_instruction(migraphx::op::dot{alpha, beta}, l1, l2, l3);

    return p;
}

migraphx::program create_conv1_int8()
{
    std::size_t n = 1, c = 3, ih = 299, iw = 299;
    migraphx::program p;
    migraphx::shape a_shape{migraphx::shape::int8_type, {n, c, ih, iw}};
    std::vector<int8_t> a(n * c * ih * iw);
    std::iota(a.begin(), a.end(), 0);
    auto al = p.add_literal(migraphx::literal{a_shape, a});
    std::size_t wh = 3, ww = 3;
    std::vector<int8_t> w(32 * 3 * wh * ww);
    std::iota(w.begin(), w.end(), 0);
    migraphx::shape w_shape{migraphx::shape::int8_type, {32, 3, wh, ww}};
    auto wl = p.add_literal(migraphx::literal{w_shape, w});

    p.add_instruction(migraphx::op::quant_convolution{{{0, 0}}, {{2, 2}}, {{1, 1}}}, al, wl);
    return p;
}


migraphx::program create_conv2_float()
{
    std::size_t n = 1, c = 3, ih = 299, iw = 299;
    migraphx::program p;
    migraphx::shape a_shape{migraphx::shape::float_type, {n, c, ih, iw}};
    std::vector<float> a(n * c * ih * iw);
    std::iota(a.begin(), a.end(), 0);
    auto al = p.add_literal(migraphx::literal{a_shape, a});
    std::size_t wh = 3, ww = 3;
    std::vector<float> w(32 * 3 * wh * ww);
    std::iota(w.begin(), w.end(), 0);
    migraphx::shape w_shape{migraphx::shape::float_type, {32, 3, wh, ww}};
    auto wl = p.add_literal(migraphx::literal{w_shape, w});

    p.add_instruction(migraphx::op::convolution{{{0, 0}}, {{2, 2}}, {{1, 1}}}, al, wl);
    return p;
}


migraphx::program create_conv2_int8()
{
    std::size_t n = 1, c = 3, ih = 299, iw = 299;
    migraphx::program p;
    migraphx::shape a_shape{migraphx::shape::int8_type, {n, c, ih, iw}};
    std::size_t wh = 3, ww = 3;
    migraphx::shape w_shape{migraphx::shape::int8_type, {32, 3, wh, ww}};

    auto a = read_input<int8_t>("arg_0.txt");
    if (a.size() != a_shape.elements())
    {
        std::cout << "input data for a is wrong!" << std::endl;
        std::abort();
    }
    auto w = read_input<int8_t>("arg_1.txt");
    if (w.size() != w_shape.elements())
    {
        std::cout << "input data for w is wrong!" << std::endl;
        std::abort();
    }

    auto al = p.add_literal(migraphx::literal{a_shape, a});
    auto wl = p.add_literal(migraphx::literal{w_shape, w});

    p.add_instruction(migraphx::op::quant_convolution{{{0, 0}}, {{2, 2}}, {{1, 1}}}, al, wl);
    return p;
}

migraphx::program create_conv3_int8()
{
    std::size_t in = 1, ic = 384, ih = 13, iw = 13;
    migraphx::program p;
    migraphx::shape a_shape{migraphx::shape::int8_type, {in, ic, ih, iw}};
    std::size_t wn = 256, wc = 384, wh = 3, ww = 3;
    migraphx::shape w_shape{migraphx::shape::int8_type, {wn, wc, wh, ww}};

    auto x = p.add_parameter("x", a_shape);
    auto w = p.add_parameter("w", w_shape);

    p.add_instruction(migraphx::op::quant_convolution{{{0, 0}}, {{2, 2}}, {{1, 1}}}, x, w);
    return p;
}

migraphx::program create_conv3_fp32()
{
    std::size_t in = 1, ic = 384, ih = 13, iw = 13;
    migraphx::program p;
    migraphx::shape a_shape{migraphx::shape::float_type, {in, ic, ih, iw}};
    std::size_t wn = 256, wc = 384, wh = 3, ww = 3;
    migraphx::shape w_shape{migraphx::shape::float_type, {wn, wc, wh, ww}};

    auto x = p.add_parameter("x", a_shape);
    auto w = p.add_parameter("w", w_shape);

    p.add_instruction(migraphx::op::convolution{{{0, 0}}, {{2, 2}}, {{1, 1}}}, x, w);
    return p;
}

//migraphx::program create_int8_gemm1()
//{
//    migraphx::program p;
//    migraphx::shape m1_shape{migraphx::shape::int8_type, {23, 4096}};
//    migraphx::shape m2_shape{migraphx::shape::int8_type, {4096, 100}};
//    migraphx::shape m3_shape{migraphx::shape::int32_type, {10, 8}};
//
//    auto l1  = p.add_parameter("a", m1_shape);
//    //auto tl1 = p.add_instruction(migraphx::op::transpose{{1, 0}}, l1);
//    auto l2  = p.add_parameter("b", m2_shape);
//    //auto tl2 = p.add_instruction(migraphx::op::transpose{{1, 0}}, l2);
//    auto l3  = p.add_parameter("c", m3_shape);
//    p.add_instruction(migraphx::op::quant_dot{}, l1, l2, l3);
//    return p;
//}

migraphx::program create_int8_gemm1()
{
    migraphx::program p;
    migraphx::shape m1_shape{migraphx::shape::int8_type, {101, 4096}};
    migraphx::shape m2_shape{migraphx::shape::int8_type, {100, 4096}};
    migraphx::shape m3_shape{migraphx::shape::int32_type, {101, 100}};

    auto l1  = p.add_parameter("a", m1_shape);
    auto l2  = p.add_parameter("b", m2_shape);
    auto tl2 = p.add_instruction(migraphx::op::transpose{{1, 0}}, l2);
    auto l3  = p.add_parameter("c", m3_shape);
    p.add_instruction(migraphx::op::quant_dot{}, l1, tl2, l3);
    return p;
}

migraphx::program create_int8_gemm2()
{
    migraphx::program p;
    migraphx::shape m1_shape{migraphx::shape::int8_type, {4096, 101}};
    migraphx::shape m2_shape{migraphx::shape::int8_type, {4096, 100}};
    migraphx::shape m3_shape{migraphx::shape::int32_type, {101, 100}};

    auto l1  = p.add_parameter("a", m1_shape);
    auto al1 = p.add_instruction(migraphx::op::add{}, l1, l1);
    auto tl1 = p.add_instruction(migraphx::op::transpose{{1, 0}}, al1);
    auto l2  = p.add_parameter("b", m2_shape);
    auto l3  = p.add_parameter("c", m3_shape);
    p.add_instruction(migraphx::op::quant_dot{}, tl1, l2, l3);

    return p;
}

migraphx::program create_float_gemm()
{
    migraphx::program p;
    migraphx::shape m1_shape{migraphx::shape::float_type, {101, 4096}};
    migraphx::shape m2_shape{migraphx::shape::float_type, {4096, 100}};
    migraphx::shape m3_shape{migraphx::shape::float_type, {101, 100}};

    auto l1  = p.add_parameter("a", m1_shape);
    auto l2  = p.add_parameter("b", m2_shape);
    auto l3  = p.add_parameter("c", m3_shape);
    p.add_instruction(migraphx::op::dot{0.31f, 0.27f}, l1, l2, l3);
    return p;
}


migraphx::program create_half_gemm()
{
    migraphx::program p;
    migraphx::shape m1_shape{migraphx::shape::half_type, {101, 4096}};
    migraphx::shape m2_shape{migraphx::shape::half_type, {4096, 100}};
    migraphx::shape m3_shape{migraphx::shape::half_type, {101, 100}};

    auto l1  = p.add_parameter("a", m1_shape);
    auto l2  = p.add_parameter("b", m2_shape);
    auto l3  = p.add_parameter("c", m3_shape);
    p.add_instruction(migraphx::op::dot{}, l1, l2);
    return p;
}

migraphx::program create_gemm_int8(int8_t alpha = 1, int8_t beta = 1)
{
    migraphx::program p;
    migraphx::shape m1_shape{migraphx::shape::int8_type, {4, 3}};
    migraphx::shape m2_shape{migraphx::shape::int8_type, {4, 5}};
    migraphx::shape m3_shape{migraphx::shape::int32_type, {3, 5}};
    std::vector<int8_t> data1(3 * 4);
    std::vector<int8_t> data2(4 * 5);
    std::vector<int8_t> data3(3 * 5, 1);
    std::iota(data1.begin(), data1.end(), 0);
    std::iota(data2.begin(), data2.end(), 0);
    //std::iota(data3.begin(), data3.end(), 0);

    auto l1 = p.add_literal(migraphx::literal{m1_shape, data1});
    auto tl1 = p.add_instruction(migraphx::op::transpose{{1, 0}}, l1);
    auto l2 = p.add_literal(migraphx::literal{m2_shape, data2});
    //auto tl2 = p.add_instruction(migraphx::op::transpose{{1, 0}}, l2);
    auto l3 = p.add_literal(migraphx::literal{m3_shape, data3});

    p.add_instruction(migraphx::op::quant_dot{alpha, beta}, tl1, l2, l3);

    return p;
}



int main(int argc, char **argv) {
    std::vector<int32_t> cpu_res, gpu_res; 
    auto p = create_gemm_int8();
    run_cpu(p, cpu_res);
    run_gpu(p, gpu_res);

    bool res = compare_results(cpu_res, gpu_res);
    std::cout << (res ? "PASSED" : "FAILED") << std::endl;

    return 0;
}

