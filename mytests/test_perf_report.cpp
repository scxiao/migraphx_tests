#include <iostream>
#include <iomanip>
#include <vector>
#include <migraphx/program.hpp>
#include <migraphx/shape.hpp>
#include <migraphx/literal.hpp>
#include <migraphx/operators.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/ref/target.hpp>
#include <migraphx/gpu/target.hpp>
#include <migraphx/gpu/hip.hpp>
#include <migraphx/manage_ptr.hpp>
#include <migraphx/type_name.hpp>
#include <migraphx/make_op.hpp>
#include "test.hpp"
#include "utilities.hpp"

migraphx::program create_program() 
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::float_type, {3, 4}};
    auto param = mm->add_parameter("0", s);
    auto t0 = mm->add_instruction(migraphx::op::transpose{{1, 0}}, param);
    auto st = mm->add_instruction(migraphx::op::softmax{1}, t0);
    auto r = mm->add_instruction(migraphx::op::dot{}, st, param);
    mm->add_return({r});

    return p;
}

int main(int argc, char **argv) {
//    std::vector<std::vector<float>> cpu_res, gpu_res;
//    migraphx::program prog = create_program();
//    run_prog(prog, migraphx::ref::target{}, gpu_res, 0);
//    run_prog(prog, migraphx::gpu::target{}, cpu_res, 0);
//    
//    bool ret = compare_results(cpu_res[0], gpu_res[0]);
//    std::cout << (ret ? "PASSED!" : "FAILED!") << std::endl;
//
    migraphx::program p;
    auto* mm = p.get_main_module();
    std::cout << "loc1" << std::endl;
    std::stringstream ss;
    auto one = mm->add_literal(1);
    auto two = mm->add_literal(2);
    mm->add_instruction(migraphx::make_op("add"), one, two);
    p.compile(migraphx::ref::target{});
    std::cout << "loc2" << std::endl;
    p.perf_report(ss, 2, {});
    std::cout << "loc3" << std::endl;

    return 0;
}

