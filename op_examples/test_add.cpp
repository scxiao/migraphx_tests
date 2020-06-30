#include "utilities.hpp"

//migraphx::program create_program()
//{
//    migraphx::program p;
//    migraphx::shape s1{migraphx::shape::float_type, {4, 5}};
//    migraphx::shape s2{migraphx::shape::float_type, {1}};
//    auto a1 = p.add_parameter("1", s1);
//    auto a2 = p.add_parameter("2", s2);
//    auto ba2 = p.add_instruction(migraphx::op::multibroadcast{s1.lens()}, a2);
//
//    p.add_instruction(migraphx::op::add{}, ba2, a1);
//
//    return p;
//}

//migraphx::program create_program()
//{
//    migraphx::program p;
//    auto x  = p.add_parameter("x", migraphx::shape{migraphx::shape::float_type, {4, 3, 3, 3}});
//    auto tx = p.add_instruction(migraphx::op::transpose{{0, 1, 3, 2}}, x);
//    auto tanhx = p.add_instruction(migraphx::op::tanh{}, tx);
//    auto r     = p.add_instruction(migraphx::op::add{}, tanhx, tanhx);
//    p.add_return({tx, r});
//
//    return p;
//}

migraphx::program slice_sin_program()
{
    migraphx::program p;
    auto l = p.add_parameter("x", migraphx::shape{migraphx::shape::float_type, {2, 2}});
    auto t = p.add_instruction(migraphx::op::slice{{1}, {1}, {2}}, l);
    p.add_instruction(migraphx::op::sin{}, t);

    return p;
}

migraphx::program rnn_forward_program()
{
    std::size_t batch_size  = 2;
    std::size_t seq_len     = 1;
    std::size_t hidden_size = 4;
    std::size_t input_size  = 3;
    std::size_t num_dirct   = 1;
    float clip              = 0.0f;

    migraphx::program p;
    migraphx::shape in_shape{migraphx::shape::float_type, {seq_len, batch_size, input_size}};
    migraphx::shape w_shape{migraphx::shape::float_type, {num_dirct, hidden_size, input_size}};
    migraphx::shape r_shape{migraphx::shape::float_type, {num_dirct, hidden_size, hidden_size}};
    migraphx::shape b_shape{migraphx::shape::float_type, {num_dirct, 2 * hidden_size}};
    migraphx::shape ih_shape{migraphx::shape::float_type, {num_dirct, batch_size, hidden_size}};

    auto seq  = p.add_parameter("seq", in_shape);
    auto w    = p.add_parameter("w", w_shape);
    auto r    = p.add_parameter("r", r_shape);
    auto bias = p.add_parameter("bias", b_shape);
    auto ih   = p.add_parameter("ih", ih_shape);
    auto und  = p.add_instruction(migraphx::op::undefined{});

    auto output =
        p.add_instruction(migraphx::op::rnn{hidden_size,
                                            {migraphx::op::tanh{}, migraphx::op::tanh{}},
                                            migraphx::op::rnn_direction::forward,
                                            clip},
                          seq,
                          w,
                          r,
                          bias,
                          und,
                          ih);
    p.add_instruction(migraphx::op::rnn_last_output{}, output);

    return p;
}

//migraphx::program sin_shape()
//{
//    migraphx::program p;
//    auto l = p.add_parameter("x", migraphx::shape{migraphx::shape::float_type, {2, 2}});
//    auto sl = p.add_instruction(migraphx::op::add{}, l, l);
//    auto t = p.add_instruction(migraphx::op::shape_of{}, sl);
//    if (t->can_eval())
//    {
//        auto arg_s = t->eval();
//        std::cout << "arg_s = " << arg_s << std::endl;
//    }
//
//    return p;
//}

migraphx::program onehot()
{
    migraphx::program p;
    migraphx::shape s_ind{migraphx::shape::int32_type, {3, 5, 6}};
    migraphx::shape s_val{migraphx::shape::float_type, {2}};
    std::size_t depth = 30;
    std::vector<int> vec_ind(s_ind.elements());
    std::srand(std::time(nullptr));
    std::generate(vec_ind.begin(), vec_ind.end(), [&]() {
        return rand() % depth;
    });
    auto l_ind = p.add_literal(migraphx::literal(s_ind, vec_ind));
    auto l_val = p.add_parameter("v", s_val);
    p.add_instruction(migraphx::op::onehot{depth, 0}, l_ind, l_val);
    return p;
};

int main()
{
    auto p = onehot();
    std::vector<float> cpu_res, gpu_res;
    run_cpu(p, cpu_res);
    run_gpu(p, gpu_res);

    bool b_res = compare_results(cpu_res, gpu_res);

    std::cout << (b_res ? "PASSED!" : "FAILED!") << std::endl;

    return 0;
}
