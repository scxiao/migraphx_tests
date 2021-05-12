#include <migraphx/shape.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/permutation.hpp>
#include <migraphx/module.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/instruction_ref.hpp>
#include "utilities.hpp"

int main()
{
    migraphx::shape s{migraphx::shape::float_type, {2, 3, 4, 5}};
    migraphx::module m;
    auto l1 = m.add_parameter("x", s);
    auto l2 = m.add_instruction(migraphx::make_op("transpose", {{"dims", {2, 3, 0, 1}}}), l1);
    auto perm = migraphx::find_permutation(l2->get_shape());
    std::cout << "perm = " << perm << std::endl;

    return 0;
}

