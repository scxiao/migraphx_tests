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
#include <migraphx/onnx.hpp>
#include "utilities.hpp"


migraphx::program create_program()
{
    migraphx::program p;
    auto* mm = p.get_main_module();

    migraphx::shape s{migraphx::shape::float_type, {2, 3}};
    auto x = mm->add_parameter("x", s);
    auto y = mm->add_parameter("y", s);

    migraphx::shape sc{migraphx::shape::bool_type, {1}};
    auto cond = mm->add_parameter("cond", sc);


    auto smod_sum = mm->create_sub_module("add");
    auto sum = smod_sum->add_instruction(migraphx::op::add{}, x, y);
    smod_sum->add_return({sum});

    auto smod_mul = mm->create_sub_module("mul");
    auto mul = smod_mul->add_instruction(migraphx::op::mul{}, x, y);
    smod_mul->add_return({mul});

    auto r = mm->add_instruction(migraphx::op::iff{"then_sub_graph", "else_sub_graph"}, {cond}, {smod_sum, smod_mul});
    mm->add_return({r});

    return p;
}

int main()
{
    auto p = create_program();
    std::cout << "p = " <<  std::endl;
    std::cout << p << std::endl;

    auto p_copy = p;
    std::cout << "p_copy = " << std::endl;
    std::cout << p_copy << std::endl;

    p.compile(migraphx::ref::target{});
    std::cout << "After compiling, p = " << std::endl;
    std::cout << p << std::endl;

    p_copy.compile(migraphx::gpu::target{});
    std::cout << "After compiling, p_copy = " << std::endl;
    std::cout << p_copy << std::endl;

    return 0;
}
