#include <iostream>
#include <vector>
#include <iomanip>
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

int main()
{
    {
		std::size_t batch_size  = 2;
		std::size_t seq_len     = 1;
		std::size_t hidden_size = 5;
		std::size_t input_size  = 3;
		std::size_t num_dirct   = 2;

		migraphx::shape w_shape{migraphx::shape::float_type, {num_dirct, 3*hidden_size, input_size}};    
		std::vector<float> w_data{
			0.3809,  0.4283,  0.2294,
			-0.1018, -0.1226, -0.0037,
			 0.2449, -0.2712, -0.1418,
			 0.1363, -0.3453, -0.0693,
			-0.2281,  0.2699, -0.2024,
			-0.3085, -0.3338,  0.4109,
			 0.2605, -0.1019, -0.2813,
			 0.3323, -0.1590,  0.0788,
			-0.3535,  0.0397,  0.2732,
			 0.2906,  0.0519,  0.3617,
			-0.2664,  0.1441,  0.0464,
			-0.1057,  0.2204, -0.3294,
			 0.3670,  0.1411,  0.3852,
			 0.3572,  0.3918,  0.0483,
			-0.3906, -0.2841, -0.2778,
			-0.4272,  0.2335, -0.1811,
			-0.3885, -0.1279,  0.1000,
			 0.0206, -0.3284, -0.0353,
			 0.1197,  0.1190,  0.3862,
			 0.0965, -0.0492,  0.2657,
			-0.1430,  0.0597,  0.1408,
			-0.0315,  0.1248,  0.0751,
			 0.3838,  0.3020,  0.0515,
			 0.2375, -0.4255,  0.1714,
			-0.0432,  0.3447, -0.2441,
			-0.3989, -0.3428, -0.4204,
			-0.4080, -0.2683, -0.0996,
			-0.1685, -0.0532, -0.1258,
			 0.1663, -0.3526, -0.3915,
			-0.1721,  0.1292, -0.2279};

		migraphx::shape r_shape{migraphx::shape::float_type, {num_dirct, 3*hidden_size, hidden_size}};
		std::vector<float> r_data{
			-0.2683,  0.0699, -0.4021, -0.1379,  0.0042,
			-0.2447,  0.4006,  0.0270, -0.0446,  0.1063,
			 0.1381,  0.1310, -0.3596,  0.3869,  0.3929,
			 0.2750,  0.0890,  0.3069, -0.1691, -0.2194,
			-0.1066,  0.3187, -0.4369, -0.0603, -0.0834,
			-0.1182, -0.2047,  0.3253, -0.2931,  0.2082,
			 0.0424,  0.1111, -0.2773, -0.0279, -0.0869,
			 0.1413, -0.4227, -0.3672,  0.4137,  0.0609,
			 0.4223, -0.4032,  0.2945,  0.3600,  0.3345,
			-0.3880, -0.0192, -0.0090, -0.2648,  0.4339,
			-0.0155,  0.4437, -0.1766,  0.1957,  0.2475,
			 0.3773, -0.2710,  0.3289, -0.2077, -0.2534,
			-0.0832, -0.1632,  0.0728,  0.2520,  0.4153,
			 0.1659, -0.4342,  0.0541,  0.1812, -0.2305,
			 0.4440,  0.0946,  0.0410, -0.4381, -0.3161,
			 0.3906, -0.3958, -0.4238,  0.1975,  0.3440,
			 0.1437, -0.0568,  0.1492, -0.4248, -0.3304,
			 0.2786, -0.1328, -0.3740, -0.3566,  0.3074,
			 0.0924,  0.2684, -0.1527,  0.1826,  0.2424,
			 0.2002,  0.3479, -0.1089,  0.3472, -0.3677,
			-0.4231, -0.0798, -0.3709,  0.3924,  0.2774,
			-0.3690, -0.0233,  0.2845,  0.1969,  0.1618,
			-0.3742, -0.3619,  0.2925, -0.1838, -0.1495,
			-0.3747,  0.0341, -0.4243, -0.0732, -0.3997,
			 0.2139,  0.2425,  0.4171, -0.3358,  0.3534,
			 0.0938, -0.0582, -0.2681, -0.4293,  0.1027,
			 0.4101,  0.2641, -0.4110, -0.1681,  0.3582,
			-0.2089,  0.0852,  0.0963,  0.3866,  0.1955,
			-0.2174,  0.1996, -0.2252,  0.1748,  0.1833,
			-0.3155,  0.2567, -0.4387,  0.3402,  0.0599};

		migraphx::shape b_shape{migraphx::shape::float_type, {num_dirct, 6*hidden_size}};
		std::vector<float> bias_data{
			-0.1582, -0.0826,  0.4008, 0.0118,  0.2511,  
			0.1900, -0.2838,  0.2549, -0.2484,  0.2363, 
			-0.4083, -0.0295, -0.1161,  0.1211, 0.2509, 
			-0.1414, -0.2628, -0.2992, 0.1517,  0.1817, 
			-0.2783,  0.3183, -0.1629, -0.3108, -0.3418, 
			0.0411,  0.2203,  0.2187, -0.2990, -0.0416,
			0.0209, -0.1024,  0.4443, -0.4420, -0.0330, 
			-0.3591, -0.2990,  0.2167, 0.1395,  0.2317,  
			0.1318,  0.1909, -0.3615,  0.1953, -0.2582, 
			-0.2217,  0.3723, 0.1458,  0.2630, -0.0377, 
			0.1754,  0.0800, -0.3964, -0.3247,  0.4219, 
			-0.0900, 0.3553,  0.2614, -0.1298, -0.1124};

			std::vector<float> input_data{
			-0.8432, -0.9887,  1.3041,
			 -2.6430, -0.3306, -0.8504};

			//std::vector<float> input_data{
			//-0.8432, -0.9887,  1.3041,
			// -2.6430, -0.3306, -0.8504,
			//-0.3933,  0.5151, -0.2951,
			// 0.0093, -1.1948, -0.1239,
			// 0.0373,  1.3211,  0.7854,
			// -0.4838, -1.0536, -0.2529};

		migraphx::shape ih_shape{migraphx::shape::float_type, {num_dirct, batch_size, hidden_size}};
		std::vector<float> ih_data{
			-0.0468,  0.5691, -0.0882,  0.8340,  0.1483, 
			-0.3902, -0.5348,  0.4178,  1.0175, 0.9212, 
			-0.0468,  0.5691, -0.0882,  0.8340,  0.1483,
			-0.3902, -0.5348,  0.4178,  1.0175,  0.9212};
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

			auto concat_hs = p.add_instruction(migraphx::op::gru{hidden_size,
												{migraphx::op::sigmoid{}, migraphx::op::tanh{}},
												migraphx::op::rnn_direction::bidirectional,
												clip,
                                                1},
							  seq,
							  w,
							  r,
							  bias,
                              und,
                              ih);
            p.add_instruction(migraphx::op::rnn_last_output{}, concat_hs);
			p.compile(migraphx::cpu::target{});

			auto hs_concat = p.eval({});
			std::vector<float> hs_data;
			hs_concat.visit([&](auto output) { hs_data.assign(output.begin(), output.end()); });

            //for_each(hs_data.begin(), hs_data.end(), [](auto &f) { std::cout << f << ", "; });
            for (auto i = 0; i < hs_data.size(); i++)
            {
                std::cout << std::setw(12) << hs_data.at(i) << ", ";
                if ((i + 1) % 5 == 0)
                    std::cout << std::endl;
            }
            std::cout << std::endl;
		}
    }

        
    return 0;
}
