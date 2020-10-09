#include <iostream>
#include <vector>
#include <migraphx/literal.hpp>
#include <migraphx/operators.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/quantization.hpp>
#include <migraphx/cpu/target.hpp>
#include <migraphx/quantization.hpp>
#include <migraphx/verify.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/onnx.hpp>
#include "utilities.hpp"

migraphx::program resize_upsample_pf()
{
    migraphx::program p;

    std::vector<float> ds = {1.0f, 1.0f, 2.0f, 3.0f};
    migraphx::shape ss{migraphx::shape::float_type, {4}};
    auto lc = p.add_literal(migraphx::literal{ss, ds});
	migraphx::shape sx{migraphx::shape::float_type, {1, 1, 2, 2}};
    auto inx = p.add_parameter("X", sx);

    p.add_parameter("rio", {migraphx::shape::float_type});

    migraphx::shape si{migraphx::shape::int32_type, {1, 1, 4, 6}};
    std::vector<int> ind = {0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 2, 2, 2, 3, 3, 3};
    auto li  = p.add_literal(migraphx::literal(si, ind));

    auto lrsp = p.add_instruction(migraphx::op::reshape{{4}}, inx);
    auto r = p.add_instruction(migraphx::op::gather{0}, lrsp, li);
    p.add_return({r});

    return p;
}

migraphx::program resize_upsample_pc()
{
    migraphx::program p;

    std::vector<float> ds = {1.0f, 1.0f, 2.0f, 1.5f};
    migraphx::shape ss{migraphx::shape::float_type, {4}};
    auto lc = p.add_literal(migraphx::literal{ss, ds});
	migraphx::shape sx{migraphx::shape::float_type, {1, 1, 2, 4}};
    auto inx = p.add_parameter("X", sx);

    p.add_parameter("rio", {migraphx::shape::float_type});

    migraphx::shape si{migraphx::shape::int32_type, {1, 1, 4, 6}};
    std::vector<int> ind = {0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 2, 2, 2, 3, 3, 3};
    auto li  = p.add_literal(migraphx::literal(si, ind));

    auto lrsp = p.add_instruction(migraphx::op::reshape{{8}}, inx);
    auto r = p.add_instruction(migraphx::op::gather{0}, lrsp, li);
    p.add_return({r});

    return p;
}

migraphx::program resize_downsample_c()
{
    migraphx::program p;

    std::vector<float> ds = {1.0f, 1.0f, 0.6f, 0.6f};
    migraphx::shape ss{migraphx::shape::float_type, {4}};
    auto lc = p.add_literal(migraphx::literal{ss, ds});

	migraphx::shape sx{migraphx::shape::float_type, {1, 1, 2, 4}};
    auto inx = p.add_parameter("X", sx);

    p.add_parameter("rio", {migraphx::shape::float_type});

    migraphx::shape si{migraphx::shape::int32_type, {1, 1, 1, 2}};
    std::vector<int> ind = {0, 2};
    auto li  = p.add_literal(migraphx::literal(si, ind));

    auto lrsp = p.add_instruction(migraphx::op::reshape{{8}}, inx);
    auto r = p.add_instruction(migraphx::op::gather{0}, lrsp, li);
    p.add_return({r});

    return p;
}

migraphx::program resize_downsample_f()
{
    migraphx::program p;

    std::vector<float> ds = {1.0f, 1.0f, 0.6f, 0.6f};
    migraphx::shape ss{migraphx::shape::float_type, {4}};
    auto lc = p.add_literal(migraphx::literal{ss, ds});

	migraphx::shape sx{migraphx::shape::float_type, {1, 1, 2, 4}};
    auto inx = p.add_parameter("X", sx);

    p.add_parameter("rio", {migraphx::shape::float_type});

    migraphx::shape si{migraphx::shape::int32_type, {1, 1, 1, 2}};
    std::vector<int> ind = {4, 7};
    auto li  = p.add_literal(migraphx::literal(si, ind));

    auto lrsp = p.add_instruction(migraphx::op::reshape{{8}}, inx);
    auto r = p.add_instruction(migraphx::op::gather{0}, lrsp, li);
    p.add_return({r});

    return p;
}

migraphx::program resize_outlen()
{
    migraphx::program p;
    std::vector<int64_t> out_len = {1, 1, 4, 6};
    migraphx::shape so{migraphx::shape::int64_type, {4}};
    p.add_literal(migraphx::literal(so, out_len));

	migraphx::shape sx{migraphx::shape::float_type, {1, 1, 2, 2}};
    auto inx = p.add_parameter("X", sx);

    p.add_parameter("rio", {migraphx::shape::float_type});
    p.add_parameter("scales", {migraphx::shape::float_type});

    migraphx::shape si{migraphx::shape::int32_type, {1, 1, 4, 6}};
    std::vector<int> ind = {0, 0, 1, 1, 1, 1, 2, 2, 3, 3, 3, 3, 2, 2, 3, 3, 3, 3, 2, 2, 3, 3, 3, 3};
    auto li  = p.add_literal(migraphx::literal(si, ind));

    auto lrsp = p.add_instruction(migraphx::op::reshape{{4}}, inx);
    auto r = p.add_instruction(migraphx::op::gather{0}, lrsp, li);
    p.add_return({r});

    return p;
}


void check_prog(const migraphx::program& p1, const migraphx::program& p2)
{
    std::cout << "p1 = " << migraphx::to_string(p1) << std::endl;
    std::cout << "p2 = " << migraphx::to_string(p2) << std::endl;
    if (p1 == p2)
    {
        std::cout << "They are the same!" << std::endl;
    }
    else
    {
        std::cout << "They are different!" << std::endl;
    }
}


int main(int argc, char **argv)
{
    if (argc != 2)
    {
        std::cout << "Usage: " << argv[0] << " onnx_file" << std::endl;
        return 0;
    }


    auto p = resize_outlen();
    std::cout << "p = " << std::endl;
    std::cout << p << std::endl;

    auto prog = migraphx::parse_onnx(argv[1]);
    std::cout << "prog = " << std::endl;
    std::cout << prog << std::endl;


    check_prog(p, prog);

    return 0;
}

