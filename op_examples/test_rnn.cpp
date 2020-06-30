#include "utilities.hpp"

migraphx::program lstm_program()
{
    std::size_t batch_size  = 3;
    std::size_t seq_len     = 8;
    std::size_t hidden_size = 5;
    std::size_t input_size  = 8;
    std::size_t num_dirct   = 2;
    float clip              = 0.0f;

    migraphx::program p;
    migraphx::shape in_shape{migraphx::shape::float_type, {seq_len, batch_size, input_size}};
    migraphx::shape w_shape{migraphx::shape::float_type,
                            {num_dirct, 4 * hidden_size, input_size}};
    migraphx::shape r_shape{migraphx::shape::float_type,
                            {num_dirct, 4 * hidden_size, hidden_size}};
    migraphx::shape b_shape{migraphx::shape::float_type, {num_dirct, 8 * hidden_size}};
    migraphx::shape ih_shape{migraphx::shape::float_type, {num_dirct, batch_size, hidden_size}};
    migraphx::shape l_shape{migraphx::shape::int64_type, {batch_size}};
    migraphx::shape ic_shape{migraphx::shape::float_type, {num_dirct, batch_size, hidden_size}};
    migraphx::shape pph_shape{migraphx::shape::float_type, {num_dirct, 3 * hidden_size}};

    auto seq  = p.add_parameter("seq", in_shape);
    auto w    = p.add_parameter("w", w_shape);
    auto r    = p.add_parameter("r", r_shape);
    auto bias = p.add_parameter("bias", b_shape);
    auto ih   = p.add_parameter("ih", ih_shape);
    auto len = p.add_literal(migraphx::literal(l_shape, {1, 2, 8}));
    auto ic   = p.add_parameter("ic", ic_shape);
    auto pph  = p.add_parameter("pph", pph_shape);

    auto output = p.add_instruction(
        migraphx::op::lstm{
            hidden_size,
            {migraphx::op::sigmoid{}, migraphx::op::tanh{}, migraphx::op::tanh{}},
            migraphx::op::rnn_direction::bidirectional,
            clip},
        seq,
        w,
        r,
        bias,
        len,
        ih,
        ic,
        pph);
    //auto last_hs = p.add_instruction(migraphx::op::rnn_last_hs_output{}, output);
    p.add_return({output});

    return p;
}

migraphx::program lstm_prog() {
    std::size_t batch_size  = 3;
    std::size_t seq_len     = 4;
    std::size_t hidden_size = 4;
    std::size_t input_size  = 3;
    std::size_t num_dirct   = 1;
    std::vector<float> w_data{
        -0.2763, -0.4715, -0.3010, -0.2306, -0.2283, -0.2656, 0.2035,  0.3570,  -0.1499, 0.4390,
        -0.1843, 0.2351,  0.3357,  0.1217,  0.1401,  0.3300,  -0.0429, 0.3266,  0.4834,  -0.3914,
        -0.1480, 0.3734,  -0.0372, -0.1746, 0.0550,  0.4177,  -0.1332, 0.4391,  -0.3287, -0.4401,
        0.1486,  0.1346,  0.1048,  -0.4361, 0.0886,  -0.3840, -0.2730, -0.1710, 0.3274,  0.0169,
        -0.4462, 0.0729,  0.3983,  -0.0669, 0.0756,  0.4150,  -0.4684, -0.2522};

    std::vector<float> r_data{
        -0.4564, -0.4432, 0.1605,  0.4387,  0.0034,  0.4116,  0.2824,  0.4775,  -0.2729, -0.4707,
        0.1363,  0.2218,  0.0559,  0.2828,  0.2093,  0.4687,  0.3794,  -0.1069, -0.3049, 0.1430,
        -0.2506, 0.4644,  0.2755,  -0.3645, -0.3155, 0.1425,  0.2891,  0.1786,  -0.3274, 0.2365,
        0.2522,  -0.4312, -0.0562, -0.2748, 0.0776,  -0.3154, 0.2851,  -0.3930, -0.1174, 0.4360,
        0.2436,  0.0164,  -0.0680, 0.3403,  -0.2857, -0.0459, -0.2991, -0.2624, 0.4194,  -0.3291,
        -0.4659, 0.3300,  0.0454,  0.4981,  -0.4706, -0.4584, 0.2596,  0.2871,  -0.3509, -0.1910,
        0.3987,  -0.1687, -0.0032, -0.1038};

    std::vector<float> bias_data{-0.0258, 0.0073,  -0.4780, -0.4101, -0.3556, -0.1017, 0.3632,
                                 -0.1823, 0.1479,  0.1677,  -0.2603, 0.0381,  0.1575,  0.1896,
                                 0.4755,  -0.4794, 0.2167,  -0.4474, -0.3139, 0.1018,  0.4470,
                                 -0.4232, 0.3247,  -0.1636, -0.1582, -0.1703, 0.3920,  0.2055,
                                 -0.4386, 0.4208,  0.0717,  0.3789};

    std::vector<float> input_data{
        -0.5516, 0.2391, -1.6951, -0.4313, -0.9730, -0.2005, 2.3930,  -0.5221, -0.1331,
        -0.0910, 1.2122, -0.1952, 0.4661,  0.6494,  2.1332,  -1.0972, 0.9816,  0.1122,
        0.3577,  1.3508, -0.5366, 1.7449,  0.5483,  -0.0701, -0.4100, -2.2344, 0.3685,
        0.4583,  2.3794, 1.0372,  -0.8887, 0.7892,  -0.4012, -0.2818, -2.3374, 1.5310};

    std::vector<float> ih_data{1.5289,
                               1.0986,
                               0.6091,
                               1.6462,
                               0.8720,
                               0.5349,
                               -0.1962,
                               -1.7416,
                               -0.9912,
                               1.2831,
                               1.0896,
                               -0.6959};

    std::vector<float> ic_data{-0.8323,
                               0.3998,
                               0.1831,
                               0.5938,
                               2.7096,
                               -0.1790,
                               0.0022,
                               -0.8040,
                               0.1578,
                               0.0567,
                               0.8069,
                               -0.5141};

    std::vector<float> pph_data{-0.8271,
                                -0.5683,
                                0.4562,
                                -1.2545,
                                1.2729,
                                -0.4082,
                                -0.4392,
                                -0.9406,
                                0.7794,
                                1.8194,
                                -0.5811,
                                0.2166};

    migraphx::shape in_shape{migraphx::shape::float_type, {seq_len, batch_size, input_size}};
    migraphx::shape w_shape{migraphx::shape::float_type, {num_dirct, 4 * hidden_size, input_size}};
    migraphx::shape r_shape{migraphx::shape::float_type, {num_dirct, 4 * hidden_size, hidden_size}};
    migraphx::shape b_shape{migraphx::shape::float_type, {num_dirct, 8 * hidden_size}};
    migraphx::shape ih_shape{migraphx::shape::float_type, {num_dirct, batch_size, hidden_size}};
    migraphx::shape ic_shape{migraphx::shape::float_type, {num_dirct, batch_size, hidden_size}};
    migraphx::shape pph_shape{migraphx::shape::float_type, {num_dirct, 3 * hidden_size}};
    float clip = 0.0f;
    migraphx::program p;
    auto seq = p.add_literal(migraphx::literal{in_shape, input_data});

    auto w = p.add_literal(migraphx::literal{w_shape, w_data});
    auto r = p.add_literal(migraphx::literal{r_shape, r_data});
    auto hs = p.add_instruction(migraphx::op::lstm{hidden_size,
                                         {migraphx::op::sigmoid{}},
                                         migraphx::op::rnn_direction::reverse,
                                         clip,
                                         0},
                      seq,
                      w,
                      r);
    p.add_return({hs});
	return p;
}

migraphx::program lstm_prog1()
{
    std::size_t batch_size  = 2;
    std::size_t seq_len     = 3;
    std::size_t hidden_size = 5;
    std::size_t input_size  = 8;
    std::size_t num_dirct   = 1;
    float clip              = 0.0f;

    migraphx::program p;
    migraphx::shape in_shape{migraphx::shape::float_type, {seq_len, batch_size, input_size}};
    migraphx::shape w_shape{migraphx::shape::float_type,
                            {num_dirct, 4 * hidden_size, input_size}};
    migraphx::shape r_shape{migraphx::shape::float_type,
                            {num_dirct, 4 * hidden_size, hidden_size}};
    migraphx::shape b_shape{migraphx::shape::float_type, {num_dirct, 8 * hidden_size}};
    migraphx::shape ih_shape{migraphx::shape::float_type, {num_dirct, batch_size, hidden_size}};
    migraphx::shape l_shape{migraphx::shape::int32_type, {batch_size}};
    migraphx::shape ic_shape{migraphx::shape::float_type, {num_dirct, batch_size, hidden_size}};
    migraphx::shape pph_shape{migraphx::shape::float_type, {num_dirct, 3 * hidden_size}};

    auto seq  = p.add_parameter("seq", in_shape);
    auto w    = p.add_parameter("w", w_shape);
    auto r    = p.add_parameter("r", r_shape);
    auto bias = p.add_parameter("bias", b_shape);
    auto ih   = p.add_parameter("ih", ih_shape);
    auto len  = p.add_literal(migraphx::literal(l_shape, {1, 2}));
    auto ic   = p.add_parameter("ic", ic_shape);
    auto pph  = p.add_parameter("pph", pph_shape);

    auto output = p.add_instruction(
        migraphx::op::lstm{
            hidden_size,
            {migraphx::op::sigmoid{}, migraphx::op::tanh{}, migraphx::op::tanh{}},
            migraphx::op::rnn_direction::forward,
            clip},
        seq,
        w,
        r,
        bias,
        len,
        ih,
        ic,
        pph);
    p.add_instruction(migraphx::op::rnn_last_hs_output{}, output);

    return p;
}

migraphx::program lstm_prog2()
{
    std::size_t batch_size  = 2;
    std::size_t seq_len     = 3;
    std::size_t hidden_size = 5;
    std::size_t input_size  = 8;
    std::size_t num_dirct   = 2;
    float clip              = 0.0f;

    migraphx::program p;
    migraphx::shape in_shape{migraphx::shape::float_type, {seq_len, batch_size, input_size}};
    migraphx::shape w_shape{migraphx::shape::float_type,
                            {num_dirct, 4 * hidden_size, input_size}};
    migraphx::shape r_shape{migraphx::shape::float_type,
                            {num_dirct, 4 * hidden_size, hidden_size}};
    migraphx::shape b_shape{migraphx::shape::float_type, {num_dirct, 8 * hidden_size}};
    migraphx::shape ih_shape{migraphx::shape::float_type, {num_dirct, batch_size, hidden_size}};
    migraphx::shape ic_shape{migraphx::shape::float_type, {num_dirct, batch_size, hidden_size}};
    migraphx::shape pph_shape{migraphx::shape::float_type, {num_dirct, 3 * hidden_size}};

    auto seq  = p.add_parameter("seq", in_shape);
    auto w    = p.add_parameter("w", w_shape);
    auto r    = p.add_parameter("r", r_shape);
    auto bias = p.add_parameter("bias", b_shape);
    auto ih   = p.add_parameter("ih", ih_shape);
    auto ic   = p.add_parameter("ic", ic_shape);
    auto pph  = p.add_parameter("pph", pph_shape);
    auto und  = p.add_instruction(migraphx::op::undefined{});

    auto output = p.add_instruction(
        migraphx::op::lstm{
            hidden_size,
            {migraphx::op::sigmoid{}, migraphx::op::tanh{}, migraphx::op::tanh{}},
            migraphx::op::rnn_direction::bidirectional,
            clip},
        seq,
        w,
        r,
        bias,
        und,
        ih,
        ic,
        pph);
    p.add_instruction(
        migraphx::op::rnn_last_hs_output{}, output);

    return p;
}

migraphx::program lstm_prog3()
{
    std::size_t batch_size  = 2;
    std::size_t seq_len     = 3;
    std::size_t hidden_size = 5;
    std::size_t input_size  = 8;
    std::size_t num_dirct   = 2;
    float clip              = 0.0f;

    migraphx::program p;
    migraphx::shape in_shape{migraphx::shape::float_type, {seq_len, batch_size, input_size}};
    migraphx::shape w_shape{migraphx::shape::float_type,
                            {num_dirct, 4 * hidden_size, input_size}};
    migraphx::shape r_shape{migraphx::shape::float_type,
                            {num_dirct, 4 * hidden_size, hidden_size}};
    migraphx::shape b_shape{migraphx::shape::float_type, {num_dirct, 8 * hidden_size}};
    migraphx::shape ih_shape{migraphx::shape::float_type, {num_dirct, batch_size, hidden_size}};

    auto seq  = p.add_parameter("seq", in_shape);
    auto w    = p.add_parameter("w", w_shape);
    auto r    = p.add_parameter("r", r_shape);
    auto bias = p.add_parameter("bias", b_shape);
    auto ih   = p.add_parameter("ih", ih_shape);
    auto und  = p.add_instruction(migraphx::op::undefined{});

    p.add_instruction(migraphx::op::lstm{hidden_size,
                                         {migraphx::op::sigmoid{}, migraphx::op::tanh{}},
                                         migraphx::op::rnn_direction::bidirectional,
                                         clip},
                      seq,
                      w,
                      r,
                      bias,
                      und,
                      ih);

    return p;
}

migraphx::program lstm_prog4()
{
    std::size_t batch_size  = 2;
    std::size_t seq_len     = 3;
    std::size_t hidden_size = 5;
    std::size_t input_size  = 8;
    std::size_t num_dirct   = 1;
    float clip              = 0.0f;

    migraphx::program p;
    migraphx::shape in_shape{migraphx::shape::float_type, {seq_len, batch_size, input_size}};
    migraphx::shape w_shape{migraphx::shape::float_type,
                            {num_dirct, 4 * hidden_size, input_size}};
    migraphx::shape r_shape{migraphx::shape::float_type,
                            {num_dirct, 4 * hidden_size, hidden_size}};
    auto seq = p.add_parameter("seq", in_shape);
    auto w   = p.add_parameter("w", w_shape);
    auto r   = p.add_parameter("r", r_shape);
    auto hs  = p.add_instruction(
        migraphx::op::lstm{
            hidden_size,
            {migraphx::op::sigmoid{}, migraphx::op::tanh{}, migraphx::op::tanh{}},
            migraphx::op::rnn_direction::forward,
            clip},
        seq,
        w,
        r);
    auto last_hs = p.add_instruction(migraphx::op::rnn_last_hs_output{}, hs);
    p.add_return({hs, last_hs});

    return p;
}


int main()
{
    auto p = lstm_prog4();
    std::cout << "p = " << p << std::endl;

    std::vector<std::vector<float>> cpu_res, gpu_res;
    run_prog(p, migraphx::cpu::target{}, cpu_res);
    run_prog(p, migraphx::gpu::target{}, gpu_res);

//    bool b_res = compare_results(cpu_res, gpu_res);

//    std::cout << (b_res ? "PASSED!" : "FAILED!") << std::endl;

    return 0;
}
