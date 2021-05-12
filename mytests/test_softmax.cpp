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
#include <migraphx/permutation.hpp>
#include "test.hpp"
#include "utilities.hpp"

migraphx::program create_program() 
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::float_type, {3, 4, 5}};
    auto param = mm->add_parameter("0", s);
    auto t0 = mm->add_instruction(migraphx::op::transpose{{1, 2, 0}}, param);
    auto ts = t0->get_shape();
    auto perm = migraphx::find_permutation(ts);
    std::cout << "perm = " << perm << std::endl;
    auto st = mm->add_instruction(migraphx::op::softmax{1}, t0);
    auto r = mm->add_instruction(migraphx::op::dot{}, st, param);
    mm->add_return({r});

    return p;
}

int main(int argc, char **argv) {
    std::vector<std::vector<float>> cpu_res, gpu_res;
    migraphx::program prog = create_program();
    run_options ref_options;
    ref_options.t = migraphx::ref::target{};
    run_prog(prog, cpu_res, ref_options);
    run_options gpu_options;
    gpu_options.t = migraphx::gpu::target{};
    run_prog(prog, gpu_res, gpu_options);
    
    bool ret = compare_results(cpu_res[0], gpu_res[0]);
    std::cout << (ret ? "PASSED!" : "FAILED!") << std::endl;

    return 0;
}

