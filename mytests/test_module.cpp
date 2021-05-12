#include <iostream>
#include <vector>
#include <string>
#include <migraphx/literal.hpp>
#include <migraphx/operators.hpp>
#include <migraphx/program.hpp>
#include <migraphx/module.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/gpu/target.hpp>
#include <migraphx/pass_manager.hpp>
#include <migraphx/normalize_ops.hpp>
#include <migraphx/dead_code_elimination.hpp>
#include <migraphx/simplify_algebra.hpp>
#include <migraphx/pass_manager.hpp>
#include <migraphx/onnx.hpp>
#include <migraphx/make_op.hpp>
#include "utilities.hpp"

void run_pass(migraphx::module& m)
{
    migraphx::run_passes(m, {migraphx::simplify_algebra{}, migraphx::dead_code_elimination{}});
}

migraphx::program create_program()
{
    migraphx::program p;
    auto* mm = p.get_main_module();

    auto x = mm->add_parameter("x", {migraphx::shape::int64_type});
    auto y = mm->add_parameter("y", {migraphx::shape::int64_type});

    auto sum = mm->add_instruction(migraphx::op::add{}, x, y);
    auto one = mm->add_literal(int64_t(1));
    mm->add_instruction(migraphx::op::add{}, sum, one);

    return p;
}


void case1()
{
    migraphx::program p1 = create_program();
    migraphx::program p2 = create_program();

    p2 = p1;
}

void module_print_graph()
{
    migraphx::program p1 = create_program();
    migraphx::program p2 = create_program();

    auto* mm1 = p1.get_main_module();
    auto* mm2 = p2.get_main_module();

    std::stringstream ss1;
    mm1->print_graph(ss1, true);
    std::cout << "str1 = " << std::endl;
    std::cout << ss1.str() << std::endl;

    std::stringstream ss2;
    mm1->print_cpp(ss2);
    std::cout << "str2 = " << std::endl;
    std::cout << ss2.str() << std::endl;
}

void module_annotate()
{
    migraphx::program p1 = create_program();
    migraphx::program p2 = create_program();

    auto* mm1 = p1.get_main_module();
    auto* mm2 = p2.get_main_module();

    std::stringstream ss1;
    mm1->annotate(ss1, [](auto ins) {
        std::cout << ins->name() << "_1" << std::endl;
    });

    std::cout << "str1 = " << std::endl;
    std::cout << ss1.str() << std::endl;

    std::stringstream ss2;
    mm1->annotate(ss2, [](auto ins) {
        std::cout << ins->name() << "_2" << std::endl;
    });
    std::cout << "str2 = " << std::endl;
    std::cout << ss2.str() << std::endl;
}

migraphx::program create_program1()
{
    migraphx::program p;
    auto* mm = p.get_main_module();

    auto x   = mm->add_parameter("x", {migraphx::shape::int32_type});
    auto two = mm->add_literal(2);
    auto add = mm->add_instruction(migraphx::make_op("add"), x, two);
    mm->add_return({add});
    return p;
}

migraphx::module split_prog()
{
    auto s = migraphx::shape{migraphx::shape::int32_type, {3, 2, 4}};
    migraphx::module m1;
    {
        auto b       = migraphx::op::broadcast{1, {3, 1, 4}};
         
        auto input = m1.add_parameter("input", s);
        auto x     = m1.add_instruction(
            migraphx::make_op("slice", {{"axes", {1}}, {"starts", {0}}, {"ends", {1}}}), input);
        auto y = m1.add_instruction(
            migraphx::make_op("slice", {{"axes", {1}}, {"starts", {1}}, {"ends", {2}}}), input);

        auto one   = m1.add_literal(1);
        auto oneb  = m1.add_instruction(b, one);
        auto two   = m1.add_literal(2);
        auto twob  = m1.add_instruction(b, two);

        auto sum1  = m1.add_instruction(migraphx::make_op("add"), x, oneb);
        auto sum2  = m1.add_instruction(migraphx::make_op("add"), y, twob);

        auto rdc1  = m1.add_instruction(migraphx::make_op("reduce_max", {{"axes", {0, 1}}}), sum1);
        auto rdc2  = m1.add_instruction(migraphx::make_op("reduce_max", {{"axes", {0, 1}}}), sum2);
        m1.add_return({rdc1, rdc2});
    }

	return m1;
}

migraphx::module split_prog1()
{
    auto s = migraphx::shape{migraphx::shape::int32_type, {3, 2, 4}};
    migraphx::module m1;
    {
        auto b       = migraphx::op::broadcast{1, {3, 1, 4}};
         
        auto input = m1.add_parameter("input", s);
        auto x     = m1.add_instruction(
            migraphx::make_op("slice", {{"axes", {1}}, {"starts", {0}}, {"ends", {1}}}), input);
        auto y = m1.add_instruction(
            migraphx::make_op("slice", {{"axes", {1}}, {"starts", {1}}, {"ends", {2}}}), input);

        auto one   = m1.add_literal(1);
        auto oneb  = m1.add_instruction(b, one);
        auto two   = m1.add_literal(2);
        auto twob  = m1.add_instruction(b, two);

        auto sum1  = m1.add_instruction(migraphx::make_op("add"), x, oneb);
        auto sum2  = m1.add_instruction(migraphx::make_op("add"), y, twob);

        auto rdc1  = m1.add_instruction(migraphx::make_op("reduce_max", {{"axes", {0, 2}}}), sum1);
        auto sqz1  = m1.add_instruction(migraphx::make_op("squeeze", {{"axes", {0, 1, 2}}}), rdc1);
        auto rdc2  = m1.add_instruction(migraphx::make_op("reduce_max", {{"axes", {0, 2}}}), sum2);
        auto sqz2  = m1.add_instruction(migraphx::make_op("squeeze", {{"axes", {0, 1, 2}}}), rdc2);
        m1.add_return({sqz1, sqz2});
    }

	return m1;
}

migraphx::module split_prog2()
{
    auto s = migraphx::shape{migraphx::shape::int32_type, {3, 2, 4}};
    migraphx::module m1;
    {
        auto b       = migraphx::op::broadcast{1, {3, 1, 4}};
         
        auto input = m1.add_parameter("input", s);
        auto x     = m1.add_instruction(
            migraphx::make_op("slice", {{"axes", {1}}, {"starts", {0}}, {"ends", {1}}}), input);
        auto y = m1.add_instruction(
            migraphx::make_op("slice", {{"axes", {1}}, {"starts", {1}}, {"ends", {2}}}), input);

        auto rx = m1.add_instruction(migraphx::make_op("relu"), x);

        auto rmax0  = m1.add_instruction(migraphx::make_op("reduce_max", {{"axes", {0, 2}}}), x);
        auto rmin0  = m1.add_instruction(migraphx::make_op("reduce_min", {{"axes", {0, 2}}}), x);
        auto rmax1  = m1.add_instruction(migraphx::make_op("reduce_max", {{"axes", {0, 2}}}), rx);
        auto rmin1  = m1.add_instruction(migraphx::make_op("reduce_min", {{"axes", {0, 2}}}), rx);
        auto rmax2  = m1.add_instruction(migraphx::make_op("reduce_max", {{"axes", {0, 2}}}), y);
        auto rmin2  = m1.add_instruction(migraphx::make_op("reduce_min", {{"axes", {0, 2}}}), y);
        m1.add_return({rmax0, rmin0, rmax1, rmin1, rmax2, rmin2});
    }

	return m1;
}

migraphx::module split_prog3()
{
    auto s = migraphx::shape{migraphx::shape::int32_type, {3, 2, 4}};
    migraphx::module m1;
    {
        auto b       = migraphx::op::broadcast{1, {3, 1, 4}};
         
        auto input = m1.add_parameter("input", s);
        auto x     = m1.add_instruction(
            migraphx::make_op("slice", {{"axes", {1}}, {"starts", {0}}, {"ends", {1}}}), input);
        auto y = m1.add_instruction(
            migraphx::make_op("slice", {{"axes", {1}}, {"starts", {1}}, {"ends", {2}}}), input);

        auto rx = m1.add_instruction(migraphx::make_op("relu"), x);

        auto rmax0  = m1.add_instruction(migraphx::make_op("reduce_max", {{"axes", {0, 2}}}), x);
        auto rmin0  = m1.add_instruction(migraphx::make_op("reduce_min", {{"axes", {0, 2}}}), x);
        auto rmax2  = m1.add_instruction(migraphx::make_op("reduce_max", {{"axes", {0, 2}}}), y);
        auto rmin2  = m1.add_instruction(migraphx::make_op("reduce_min", {{"axes", {0, 2}}}), y);
        m1.add_return({rx, rmax0, rmin0, rmax2, rmin2});
    }

	return m1;
}

migraphx::program test_copy()
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape sd{migraphx::shape::float_type, {2, 3}};
    auto x = mm->add_parameter("x", sd);
    migraphx::shape s_cond{migraphx::shape::bool_type, {1}};
    auto cond = mm->add_parameter("cond", s_cond);

    std::vector<float> one(sd.elements(), 1);
    std::vector<float> two(sd.elements(), 2);

    auto* then_smod = p.create_module("then_smod");
    auto l1 = then_smod->add_literal(migraphx::literal{sd, one});
    auto r1 = then_smod->add_instruction(migraphx::make_op("add"), x, l1);
    then_smod->add_return({r1});
    std::cout << "then_size = " << then_smod->size() << std::endl;
    std::cout << "then_out_shape = " << then_smod->get_output_shapes()[0] << std::endl;
    std::cout << "then_pointer = " << then_smod << std::endl;

    auto* else_smod = p.create_module("else_smod");
    auto l2 = else_smod->add_literal(migraphx::literal{sd, two});
    auto r2 = else_smod->add_instruction(migraphx::make_op("mul"), x, l2);
    else_smod->add_return({r2});
    std::cout << "else_size = " << else_smod->size() << std::endl;
    std::cout << "else_out_shape = " << else_smod->get_output_shapes()[0] << std::endl;
    std::cout << "else_pointer = " << else_smod << std::endl;

    auto ret = mm->add_instruction(migraphx::make_op("iff", {{"then_sub_graph", "then_smod"}, {"else_sub_graph",
                "else_smod"}}), {cond}, {then_smod, else_smod});
    mm->add_return({ret});
	
	return p;
}

migraphx::module module_copy()
{
    migraphx::module mm("main");
    auto x = mm.add_parameter("x", {migraphx::shape::int64_type});

    migraphx::module sm("sub");
    sm.add_instruction(migraphx::make_op("sin"), x);

    mm.add_instruction(migraphx::make_op("iff"), {x}, {&sm});

    auto mm1 = mm;

    return mm1;
}

migraphx::program module_assign()
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape sd{migraphx::shape::float_type, {2, 3}};
    auto x = mm->add_parameter("x", sd);

    std::vector<float> one(sd.elements(), 1);
    std::vector<float> two(sd.elements(), 2);

    auto* then_smod = p.create_module("then_smod");
    auto l1         = then_smod->add_literal(migraphx::literal{sd, one});
    auto r1         = then_smod->add_instruction(migraphx::make_op("add"), x, l1);
    then_smod->add_return({r1});

    auto* else_smod = p.create_module("else_smod");
    auto l2         = else_smod->add_literal(migraphx::literal{sd, two});
    auto r2         = else_smod->add_instruction(migraphx::make_op("mul"), x, l2);
    else_smod->add_return({r2});

    migraphx::shape s_cond{migraphx::shape::bool_type, {1}};
    auto cond = mm->add_parameter("cond", s_cond);
    auto ret  = mm->add_instruction(migraphx::make_op("if"), {cond}, {then_smod, else_smod});
    mm->add_return({ret});

	return p;
}

migraphx::module create_module()
{
    migraphx::module m("abc");
    migraphx::shape s{migraphx::shape::float_type, {2, 3}};
    std::vector<float> data = {0, 1, 2, 3, 4, 5};
    auto pm = m.add_parameter("x", s);
    auto lt = m.add_literal(migraphx::literal(s, data));
    auto r = m.add_instruction(migraphx::make_op("add"), pm, lt);
    m.add_return({r});

    return m;
}


int main()
{
    auto m = create_module();
    std::cout << "p = " << std::endl;
    std::cout << m << std::endl;

    auto m1 = m;
    std::cout << "p1 = " << std::endl;
    std::cout << m1 << std::endl;
    //migraphx::module m1 = split_prog2();
    //std::cout << "p1 = " << m1 << std::endl;

	//run_pass(m1);
    //std::cout << "p1 = " << m1 << std::endl;

    return 0;
}

