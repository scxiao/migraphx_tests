#include <iostream>
#include <vector>
#include <string>
#include <migraphx/literal.hpp>
#include <migraphx/operators.hpp>
#include <migraphx/program.hpp>
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

    migraphx::shape s{migraphx::shape::float_type, {30522, 768}};
    std::vector<float> data(s.elements());
    std::iota(data.begin(), data.end(), 0.5);
    auto a0 = p.add_literal(migraphx::literal{s, data});
    migraphx::shape s_indices{migraphx::shape::int32_type, {1, 11}};
    std::vector<int> indices(s_indices.elements(), 1);
    auto a1 = p.add_literal(migraphx::literal{s_indices, indices});
    auto g = p.add_instruction(migraphx::op::gather{0}, a0, a1);

    return p;
}

migraphx::program reduce_program()
{ 
    migraphx::program p;

    migraphx::shape s{migraphx::shape::float_type, {2, 8, 4, 8}};
    auto a1 = p.add_parameter("s", s);
    auto g = p.add_instruction(migraphx::op::reduce_sum{{1, 2}}, a1);
    //auto g = p.add_instruction(migraphx::op::flatten{}, a1);
    p.add_return({g});

    return p;
}


void gather_test()
{
    migraphx::program p;

    std::vector<float> data(3 * 3);
    std::iota(data.begin(), data.end(), 0.5);
    migraphx::shape s{migraphx::shape::float_type, {3, 3}};
    auto a0 = p.add_literal(migraphx::literal{s, data});
    migraphx::shape s_indices{migraphx::shape::int32_type, {1, 2}};
    std::vector<int> indices{0, 2};
    auto a1  = p.add_literal(migraphx::literal{s_indices, indices});
    int axis = -1;
    p.add_instruction(migraphx::op::gather{axis}, a0, a1);
    p.compile(migraphx::cpu::target{});
    auto result = p.eval({}).back();
    std::vector<float> res_data(4 * 5);
    std::vector<float> golden = {0.5f, 2.5f, 3.5f, 5.5f, 6.5f, 8.5f};
    result.visit([&](auto output) { res_data.assign(output.begin(), output.end()); });

}

struct normalize_test_op
{
    std::vector<int64_t> axes = {};

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return migraphx::pack(f(self.axes, "axes"));
    }

    migraphx::value attributes() const
    {
        migraphx::value attr;
        attr["axes"] = migraphx::value::array{migraphx::op::op_normalize_attributes::clip_max, 
                                            migraphx::op::op_normalize_attributes::clip_min};
        return {{"normalize_axes", attr}};
    }

    std::string name() const { return "normalize_ops_test::test_op"; }
    migraphx::shape normalize_compute_shape(std::vector<migraphx::shape> inputs) const
    {
        return inputs[0];
    }
    migraphx::argument compute(migraphx::context&,
                               const migraphx::shape& output_shape,
                               const std::vector<migraphx::argument>&) const
    {
        return {output_shape};
    }
};

void run_pass(migraphx::program& p)
{
    migraphx::run_passes(p, {migraphx::normalize_ops{}, migraphx::dead_code_elimination{}});
}

migraphx::program create_test_op(const std::vector<int64_t>& axes)
{
    migraphx::program p;
    migraphx::shape sd{migraphx::shape::float_type, {2, 3, 4}};
    auto di = p.add_parameter("data", sd);
    auto r  = p.add_instruction(normalize_test_op{axes}, di);
    p.add_return({r});

    return p;
}

void test_op()
{
    std::vector<int64_t> axes1 = {-4, 5};
    auto p1 = create_test_op(axes1);

    std::vector<int64_t> axes2 = {1, 2};
    auto p2 = create_test_op(axes2);

    run_pass(p1);

    std::cout << "p1 = " << p1 << std::endl;
    std::cout << "p2 = " << p2 << std::endl;
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
//	auto p = reduce_program();
    test_op();

    return 0;
}
