#include "utilities.hpp"
#include <migraphx/rewrite_batchnorm.hpp>

migraphx::program create_program()
{
    const size_t d1       = 2;
    const size_t d2       = 3;
    const size_t d3       = 2;
    const size_t channels = 2;
    const size_t batches  = 3;

        migraphx::program p;

        migraphx::shape s{migraphx::shape::float_type, {batches, channels, d1, d2}};
        migraphx::shape vars{migraphx::shape::float_type, {channels, d1, d2}};
        auto x        = p.add_parameter("x", s);
        auto scale    = p.add_literal(migraphx::abs(migraphx::generate_literal(vars, 1)));
        auto bias     = p.add_literal(migraphx::abs(migraphx::generate_literal(vars, 2)));
        auto mean     = p.add_literal(migraphx::abs(migraphx::generate_literal(vars, 3)));
        auto variance = p.add_literal(migraphx::abs(migraphx::generate_literal(vars, 4)));
        p.add_instruction(
            migraphx::op::batch_norm_inference{1.0e-6, 0.9f, migraphx::op::batch_norm_inference::per_activation},
            //migraphx::op::batch_norm_inference{1.0e-6, 0.9f},
            x,
            scale,
            bias,
            mean,
            variance);
   return p;
}

void rewrite_batch() {
    migraphx::shape xs{migraphx::shape::float_type, {1, 3, 8, 8}};
    migraphx::shape ws{migraphx::shape::float_type, {4, 3, 1, 1}};
    migraphx::shape vars{migraphx::shape::float_type, {4}};

    auto create_program = [&]() {
        migraphx::program p;
        auto reshape = [&](auto ins) {
            return p.add_instruction(migraphx::op::reshape{{1, 4, 1, 1}}, ins);
        };

        auto x        = p.add_literal(migraphx::generate_literal(xs, 1));
        auto w        = p.add_literal(migraphx::generate_literal(ws, 1));
        auto conv     = p.add_instruction(migraphx::op::convolution{}, x, w);
        auto scale    = (p.add_literal(migraphx::abs(migraphx::generate_literal(vars, 1))));
        auto bias     = (p.add_literal(migraphx::abs(migraphx::generate_literal(vars, 2))));
        auto mean     = (p.add_literal(migraphx::abs(migraphx::generate_literal(vars, 3))));
        auto variance = (p.add_literal(migraphx::abs(migraphx::generate_literal(vars, 4))));
        p.add_instruction(migraphx::op::batch_norm_inference{}, conv, scale, bias, mean, variance);
        return p;
    };

    migraphx::program p1 = create_program();
    migraphx::program p2 = create_program();
    migraphx::rewrite_batchnorm opt;
    opt.apply(p2);
}

int main()
{
    rewrite_batch();
//    auto p = create_program();
//    std::cout << "p = " << std::endl;
//    std::cout << p << std::endl;
//    p.compile(migraphx::gpu::target{});
//    std::cout << "p = " << std::endl;
//    std::cout << p << std::endl;
//    std::vector<std::vector<float>> res;
//    run_prog(p, migraphx::gpu::target{}, res);

    return 0;
}
