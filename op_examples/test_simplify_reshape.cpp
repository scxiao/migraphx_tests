#include <iostream>
#include <vector>
#include <migraphx/literal.hpp>
#include <migraphx/operators.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/quantization.hpp>
#include <migraphx/cpu/target.hpp>
#include <migraphx/verify.hpp>
#include <migraphx/simplify_reshapes.hpp>
#include "utilities.hpp"
#include <migraphx/pass_manager.hpp>


void run_pass(migraphx::program& p)
{
    migraphx::run_passes(p, {migraphx::simplify_reshapes{}});
}


migraphx::program create_program()
{
    migraphx::program p;
    auto s      = migraphx::shape{migraphx::shape::float_type, {1, 2, 7, 24}};
    auto x      = p.add_parameter("x", s);
    auto y      = p.add_parameter("y", s);
    auto xt     = p.add_instruction(migraphx::op::transpose{{0, 2, 3, 1}}, x);
    auto yt     = p.add_instruction(migraphx::op::transpose{{0, 2, 1, 3}}, y);
    auto concat = p.add_instruction(migraphx::op::concat{3}, xt, yt);
    //auto t      = p.add_instruction(migraphx::op::transpose{{0, 2, 3, 1}}, concat);
    p.add_return({concat});

	return p;
}

int main()
{
    auto p = create_program();

	std::cout << "p = " << std::endl;
    std::cout << p << std::endl;
    run_pass(p);	

	std::cout << "p = " << std::endl;
    std::cout << p << std::endl;

    return 0;
}

