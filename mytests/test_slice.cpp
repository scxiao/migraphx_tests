#include <iostream>
#include <vector>
#include <string>
#include <migraphx/literal.hpp>
#include <migraphx/operators.hpp>
#include <migraphx/program.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/gpu/target.hpp>
#include <migraphx/onnx.hpp>
#include "utilities.hpp"


//migraphx::program create_program()
//{ 
//    migraphx::program p;
//
//    migraphx::shape s{migraphx::shape::float_type, {30522, 768}};
//    std::vector<float> data(s.elements());
//    std::iota(data.begin(), data.end(), 0.5);
//    auto a0 = p.add_literal(migraphx::literal{s, data});
//    migraphx::shape s_indices{migraphx::shape::int32_type, {1, 11}};
//    std::vector<int> indices(s_indices.elements(), 1);
//    auto a1 = p.add_literal(migraphx::literal{s_indices, indices});
//    auto g = p.add_instruction(migraphx::op::gather{0}, a0, a1);
//
//    return p;
//}
//
//void gather_test()
//{
//    migraphx::program p;
//
//    std::vector<float> data(3 * 3);
//    std::iota(data.begin(), data.end(), 0.5);
//    migraphx::shape s{migraphx::shape::float_type, {3, 3}};
//    auto a0 = p.add_literal(migraphx::literal{s, data});
//    migraphx::shape s_indices{migraphx::shape::int32_type, {1, 2}};
//    std::vector<int> indices{0, 2};
//    auto a1  = p.add_literal(migraphx::literal{s_indices, indices});
//    int axis = -1;
//    p.add_instruction(migraphx::op::gather{axis}, a0, a1);
//    p.compile(migraphx::cpu::target{});
//    auto result = p.eval({}).back();
//    std::vector<float> res_data(4 * 5);
//    std::vector<float> golden = {0.5f, 2.5f, 3.5f, 5.5f, 6.5f, 8.5f};
//    result.visit([&](auto output) { res_data.assign(output.begin(), output.end()); });
//
//    return p;
//}

//auto create_concat = [] {
//    migraphx::program p;
//    auto a1          = p.add_instruction(allocate{create_shape(2, 2)});
//    auto p1          = p.add_instruction(simple_op{}, a1);
//    auto a2          = p.add_instruction(allocate{create_shape(2, 2)});
//    auto p2          = p.add_instruction(simple_op{}, a2);
//    std::size_t axis = -1;
//    auto a3          = p.add_instruction(allocate{create_shape(4, 2)});
//    p.add_instruction(concat(axis), p1, p2, a3);
//    return p;
//};

migraphx::program slice_program() {
    migraphx::shape input{migraphx::shape::int32_type, {2, 2, 3}};
	migraphx::program p;
	auto ins = p.add_parameter("s", input);
    auto r = p.add_instruction(migraphx::op::slice{{2}, {1}, {3}}, ins);
    p.add_return({r});

    return p;
}

int main()
{

//    auto prog = create_program();
//    auto shape = prog.get_output_shapes();
//    std::vector<float> cpu_res, gpu_res;
//    run_cpu(prog, cpu_res);
//    run_gpu(prog, gpu_res);
//    bool ret2 = compare_results(cpu_res, gpu_res);
//    std::cout << (ret2 ? "PASSED!" : "FAILED") << std::endl;
	auto p = slice_program();
    std::cout << "p = " << p << std::endl;

    return 0;
}
