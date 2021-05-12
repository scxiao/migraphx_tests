#include <iostream>
#include <vector>
#include <string>
#include <cstdlib>
#include <migraphx/make_op.hpp>
#include <migraphx/literal.hpp>
#include <migraphx/operators.hpp>
#include <migraphx/program.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/gpu/target.hpp>
#include <migraphx/onnx.hpp>
#include "utilities.hpp"

migraphx::program create_program() {
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape cond_s{migraphx::shape::bool_type};
    auto cond = mm->add_parameter("cond", cond_s);
    migraphx::shape ds{migraphx::shape::float_type, {2, 3}};
    auto x                   = mm->add_parameter("x", ds);
    auto y                   = mm->add_parameter("y", ds);
    std::vector<float> data2 = {-0.258047, 0.360394, 0.536804, -0.577762, 1.0217, 1.02442};
    auto l2                  = mm->add_literal(migraphx::literal(ds, data2));
    auto sum                 = mm->add_instruction(migraphx::make_op("add"), x, l2);

    auto* then_mod           = p.create_module("If_0_if");
    std::vector<float> data1 = {0.384804, -1.77948, -0.453775, 0.477438, -1.06333, -1.12893};
    auto l1                  = then_mod->add_literal(migraphx::literal(ds, data1));
    auto tx                  = then_mod->add_parameter("x", ds);
    auto a1                  = then_mod->add_instruction(migraphx::make_op("add"), tx, l1);
    then_mod->add_return({a1});

    auto* else_mod = p.create_module("If_0_else");
    auto ey        = else_mod->add_parameter("y", ds);
    auto a2        = else_mod->add_instruction(migraphx::make_op("mul"), ey, sum);
    else_mod->add_return({a2});

    auto ret = mm->add_instruction(migraphx::make_op("if"), {cond, x, y}, {then_mod, else_mod});
    mm->add_return({ret});

    return p;
}

auto run_prog(bool cond) {
    auto p = create_program();
    p.compile(migraphx::ref::target());
    std::vector<char> c_data = {static_cast<char>(cond)};
    migraphx::shape cs{migraphx::shape::bool_type};
    migraphx::parameter_map m;
    m["cond"] = migraphx::argument(cs, c_data.data());
    migraphx::shape ds{migraphx::shape::float_type, {2, 3}};
    std::vector<float> data_x(ds.elements(), 1);
    m["x"] = migraphx::argument(ds, data_x.data());
    std::vector<float> data_y(ds.elements(), 2);
    m["y"] = migraphx::argument(ds, data_y.data());

    auto res = p.eval(m).back();
    std::vector<float> ret;
    res.visit([&](auto v) { ret.assign(v.begin(), v.end()); });
    std::cout << std::endl;

    return ret;
}

migraphx::program create_program1()
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape cond_s{migraphx::shape::bool_type};
    auto cond = mm->add_parameter("cond", cond_s);

    migraphx::shape s{migraphx::shape::float_type, {5}};

    auto* then_mod           = p.create_module("If_0_if");
    std::vector<float> data1 = {1, 2, 3, 4, 5};
    auto l1                  = then_mod->add_literal(migraphx::literal(s, data1));
    then_mod->add_return({l1});

    auto* else_mod           = p.create_module("If_0_else");
    std::vector<float> data2 = {5, 4, 3, 2, 1};
    auto l2                  = else_mod->add_literal(migraphx::literal(s, data2));
    else_mod->add_return({l2});

    auto ret = mm->add_instruction(migraphx::make_op("if"), {cond}, {then_mod, else_mod});
    auto r   = mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), ret);
    mm->add_return({r});

    return p;
}


int main()
{
    auto p = create_program1();
    p.compile(migraphx::gpu::target{});
    std::cout << "p = " << p << std::endl;

    return 0;
}

