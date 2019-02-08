#include <iostream>
#include <vector>
#include <migraphx/program.hpp>
#include <migraphx/literal.hpp>
#include <migraphx/operators.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/cpu/target.hpp>
#include <migraphx/gpu/target.hpp>
#include <migraphx/gpu/hip.hpp>
#include <migraphx/manage_ptr.hpp>
#include <migraphx/type_name.hpp>
#include "test.hpp"

template <class T>
auto get_hash(const T& x)
{
    return std::hash<T>{}(x);
}

int main()
{
    {
		std::size_t batch_size  = 2;
		std::size_t seq_len     = 3;
		std::size_t hidden_size = 5;
		std::size_t input_size  = 3;
		std::size_t num_dirct   = 1;
		std::vector<float> w_data{0.3485, -0.0378, -0.1782,
         0.1416, -0.3096, -0.2212,
        -0.3883,  0.1983, -0.2418,
         0.1480, -0.3255,  0.1359,
        -0.3551, -0.3605, -0.3482,
        -0.1424, -0.0495, -0.1640,
        -0.1979, -0.2577, -0.4097,
        -0.1211, -0.0412,  0.1801,
         0.1721, -0.4327, -0.0498,
         0.2628, -0.1573, -0.1577,
         0.2759, -0.2023, -0.1185,
        -0.2136,  0.1294, -0.2331,
         0.0701,  0.4316,  0.0480,
         0.0247, -0.0166, -0.2729,
         0.1712, -0.3984, -0.3905};

		std::vector<float> r_data{0.2848, -0.2851, -0.3466, -0.1718, -0.1492,
        -0.0082,  0.2452, -0.0401,  0.3399,  0.2529,
        -0.0953, -0.0903, -0.1518, -0.1373,  0.3848,
        -0.0130, -0.4339,  0.0406, -0.1926, -0.1131,
         0.4285, -0.0013,  0.2243,  0.2752,  0.1776,
        -0.1720,  0.0822, -0.0295,  0.1062, -0.2721,
        -0.2736, -0.1826,  0.3541, -0.4259,  0.2188,
         0.0706,  0.3650,  0.3947,  0.2522,  0.2179,
        -0.0744,  0.2122, -0.4346,  0.2760,  0.4076,
         0.1183, -0.1500, -0.1704,  0.3090, -0.0706,
        -0.2442,  0.3021,  0.1680,  0.0783, -0.3754,
        -0.3469, -0.2972, -0.0170,  0.4143,  0.3801,
         0.3852, -0.1170, -0.2937,  0.2979, -0.1357,
         0.4257,  0.3884, -0.2916,  0.1071,  0.0934,
         0.3645, -0.4310, -0.3480,  0.0702, -0.1558};

		std::vector<float> bias_data{0.0560,  0.0310, -0.1669, -0.0781,  0.1793, -0.1758,  0.3173, -0.1650,
        -0.3732,  0.2946, -0.0912,  0.3118,  0.1391,  0.2755,  0.2695, -0.1059,
        -0.2357,  0.3629, -0.2534, -0.0494,  0.0556,  0.0881, -0.2592,
        -0.2213,  0.2310, -0.4044,  0.1801,  0.1438,  0.3108, -0.3607};

		//std::vector<float> input_data{
        //-0.8432, -0.9887,  1.3041,
        // -2.6430, -0.3306, -0.8504};

		std::vector<float> input_data{
        -0.8432, -0.9887,  1.3041,
         -2.6430, -0.3306, -0.8504,
        -0.3933,  0.5151, -0.2951,
         0.0093, -1.1948, -0.1239,
         0.0373,  1.3211,  0.7854,
         -0.4838, -1.0536, -0.2529};


        std::vector<float> ih_data{-0.0468,  0.5691, -0.0882,  0.8340,  0.1483, -0.3902, -0.5348,  0.4178,  1.0175,  0.9212};
        //std::vector<float> ih_data(batch_size * hidden_size, 0.0f);
		float clip          = 0.0f;
		{
			migraphx::program p;
			migraphx::shape in_shape{migraphx::shape::float_type, {seq_len, batch_size, input_size}};
			auto seq = p.add_literal(migraphx::literal{in_shape, input_data});

			migraphx::shape ih_shape{migraphx::shape::float_type, {num_dirct, batch_size, hidden_size}};
			auto ih = p.add_literal(migraphx::literal{ih_shape, ih_data});

			migraphx::shape w_shape{migraphx::shape::float_type, {num_dirct, 3*hidden_size, input_size}};
			auto w = p.add_literal(migraphx::literal{w_shape, w_data});

			migraphx::shape r_shape{migraphx::shape::float_type, {num_dirct, 3*hidden_size, hidden_size}};
			auto r = p.add_literal(migraphx::literal{r_shape, r_data});

			migraphx::shape b_shape{migraphx::shape::float_type, {num_dirct, 6*hidden_size}};
			auto bias = p.add_literal(migraphx::literal{b_shape, bias_data});
            auto und = p.add_instruction(migraphx::op::undefined{});

			p.add_instruction(migraphx::op::gru{hidden_size,
												{migraphx::op::sigmoid{}, migraphx::op::tanh{}},
												migraphx::op::gru::reverse,
												clip,
                                                0},
							  seq,
							  w,
							  r,
							  bias,
                              und,
                              ih);
			p.compile(migraphx::cpu::target{});

			auto hs_concat = p.eval({});
			std::vector<float> hs_data;
			hs_concat.visit([&](auto output) { hs_data.assign(output.begin(), output.end()); });

            for_each(hs_data.begin(), hs_data.end(), [](auto &f) { std::cout << f << ", "; });
            std::cout << std::endl;
		}
    }

        
    return 0;
}
