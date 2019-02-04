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

void run_cpu(migraphx::program &p)
{
    p.compile(migraphx::cpu::target{});

    migraphx::program::parameter_map m;
    for (auto &&x : p.get_parameter_shapes())
    {
        //m[x.first] = migraphx::generate_argument(x.second, get_hash(x.first));
        //std::cout << x.first << " = " << m[x.first] << std::endl;
        //if (x.first == std::string("ih")) {
        //    std::vector<float> ih_data(x.second.elements(), 0.0);
        //    m[x.first] = migraphx::argument{x.second, &ih_data[0]};
        //    //auto&& argu = migraphx::generate_argument(x.second, get_hash(x.first));
        //    //std::cout << x.first << " = " << argu << std::endl;
        //}
        //else 
        {
            auto &&argu = migraphx::generate_argument(x.second, get_hash(x.first));
            //m[x.first] = migraphx::gpu::to_gpu(migraphx::generate_argument(x.second, get_hash(x.first)));
            if (x.first == std::string("ih"))
                std::cout << x.first << " = " << argu << std::endl;
            m[x.first] = argu;
        }

    }
    auto result = p.eval(m);
    std::vector<float> resData;
    result.visit([&](auto output) { resData.assign(output.begin(), output.end()); });

    std::cout << "cpu res = " << std::endl;
    for_each(resData.begin(), resData.end(), [](float& i) { std::cout << i << "\t"; });
    std::cout << std::endl;
}

void run_gpu(migraphx::program &p)
{
    p.compile(migraphx::gpu::target{});

    migraphx::program::parameter_map m;
    for (auto &&x : p.get_parameter_shapes())
    {
        //if (x.first == std::string("ih")) {
        //    std::vector<float> ih_data(x.second.elements(), 0.0);
        //    auto && argu = migraphx::argument{x.second, &ih_data[0]};
        //    //auto&& argu = migraphx::generate_argument(x.second, get_hash(x.first));
        //    //std::cout << x.first << " = " << argu << std::endl;
        //    m[x.first] = migraphx::gpu::to_gpu(argu);
        //}
        //else 
        {
            auto&& argu = migraphx::generate_argument(x.second, get_hash(x.first));
            //m[x.first] = migraphx::gpu::to_gpu(migraphx::generate_argument(x.second, get_hash(x.first)));
            if (x.first == std::string("ih"))
                std::cout << x.first << " = " << argu << std::endl;
            m[x.first] = migraphx::gpu::to_gpu(argu);
        }
    }
    auto result = migraphx::gpu::from_gpu(p.eval(m));
    std::vector<float> resData;
    result.visit([&](auto output) { resData.assign(output.begin(), output.end()); });

    std::cout << "gpu res = " << std::endl;
    for_each(resData.begin(), resData.end(), [](float& i) { std::cout << i << "\t"; });
    std::cout << std::endl;
}

migraphx::program create_program()
{
    std::size_t batch_size  = 2;
    std::size_t seq_len     = 1;
    std::size_t hidden_size = 4;
    std::size_t input_size  = 3;
    std::size_t num_dirct   = 2;
    float clip = 0.0f;

    migraphx::program p;
    migraphx::shape in_shape{migraphx::shape::float_type, {seq_len, batch_size, input_size}};
    migraphx::shape ih_shape{migraphx::shape::float_type, {num_dirct, batch_size, hidden_size}};
    migraphx::shape w_shape{migraphx::shape::float_type, {num_dirct, hidden_size, input_size}};
    migraphx::shape r_shape{migraphx::shape::float_type, {num_dirct, hidden_size, hidden_size}};
    migraphx::shape b_shape{migraphx::shape::float_type, {num_dirct, 2 * hidden_size}};
    //std::vector<float> ih_data(ih_shape.elements(), 0.0f);
    std::vector<float> ih_data{-0.3125, 0.75, -0.75, -0.5, 0.9375, -0.9375, 0.625, 0.25, -0.0625, -1, -0.875, -0.625,
        -0.5, 0.6875, 0.3125, -0.8125};
    //std::vector<float> ih_data{-0.3125, 0.75, -0.75, -0.5, 0.9375, -0.9375, 0.625, 0.25};

    auto seq = p.add_parameter("seq", in_shape);
    auto w = p.add_parameter("w", w_shape);
    auto r = p.add_parameter("r", r_shape);
    auto bias = p.add_parameter("bias", b_shape);
    auto ih = p.add_parameter("ih", ih_shape);
    auto und = p.add_instruction(migraphx::op::undefined{});
    //auto ih = p.add_literal(migraphx::literal{ih_shape, ih_data});

    auto output = p.add_instruction(migraphx::op::rnn{hidden_size,
                                        {migraphx::op::tanh{}, migraphx::op::tanh{}},
                                        migraphx::op::rnn::bidirectional,
                                        //migraphx::op::rnn::forward,
                                        clip},
                      seq,
                      w,
                      r,
                      bias,
                      und);
                      //ih);

    p.add_instruction(migraphx::op::rnn_last_output{}, output);
    return p;
}

int main()
{
    // gpu
    auto p2 = create_program();
    run_gpu(p2);
    auto p1 = create_program();
    run_cpu(p1);
 

//    {
//		std::size_t batch_size  = 2;
//		std::size_t seq_len     = 1;
//		std::size_t hidden_size = 4;
//		std::size_t input_size  = 3;
//		std::size_t num_dirct   = 2;
//		std::vector<float> wf_data{0.4691,
//								   0.3185,
//								   -0.2227,
//								   0.4423,
//								   -0.0609,
//								   -0.2803,
//								   0.1744,
//								   0.3146,
//								   0.4049,
//								   -0.3973,
//								   -0.0890,
//								   -0.1636};
//		std::vector<float> wr_data{-0.0296,
//								   -0.1341,
//								   0.1761,
//								   -0.2325,
//								   -0.0717,
//								   0.1852,
//								   0.2720,
//								   0.1471,
//								   -0.1097,
//								   0.3363,
//								   -0.0587,
//								   -0.2302};
//		std::vector<float> rf_data{-0.0456,
//								   0.1061,
//								   0.1574,
//								   -0.4928,
//								   -0.4300,
//								   -0.1909,
//								   -0.0225,
//								   -0.2668,
//								   0.1840,
//								   -0.4453,
//								   -0.4896,
//								   0.1302,
//								   -0.0929,
//								   0.3545,
//								   -0.4981,
//								   0.0616};
//		std::vector<float> rr_data{0.2528,
//								   -0.2333,
//								   0.3973,
//								   0.1593,
//								   -0.0388,
//								   0.1702,
//								   0.3829,
//								   -0.0712,
//								   -0.1668,
//								   0.3074,
//								   -0.2854,
//								   0.4049,
//								   -0.3737,
//								   -0.1051,
//								   0.4482,
//								   -0.2841};
//		std::vector<float> biasf_data{
//			-0.4938, 0.4355, -0.3186, 0.2094, 0.1037, -0.1071, 0.4504, -0.3990};
//		std::vector<float> biasr_data{-0.3188, 0.1341, -0.4446, 0.1389, 0.3117, 0.3664, 0.2352, 0.2552};
//		std::vector<float> input(seq_len * batch_size * input_size, 0);
//		input[0] = input[1] = 1.0;
//		float clip          = 0.0f;
//		{
//			std::vector<float> ih_data(num_dirct * batch_size * hidden_size, 0);
//
//			migraphx::program p;
//			migraphx::shape in_shape{migraphx::shape::float_type, {seq_len, batch_size, input_size}};
//			auto seq = p.add_literal(migraphx::literal{in_shape, input});
//
//			migraphx::shape ih_shape{migraphx::shape::float_type, {num_dirct, batch_size, hidden_size}};
//			auto ih = p.add_literal(migraphx::literal{ih_shape, ih_data});
//
//			auto w_data = wf_data;
//			w_data.insert(w_data.end(), wr_data.begin(), wr_data.end());
//			migraphx::shape w_shape{migraphx::shape::float_type, {num_dirct, hidden_size, input_size}};
//			auto w = p.add_literal(migraphx::literal{w_shape, w_data});
//
//			auto r_data = rf_data;
//			r_data.insert(r_data.end(), rr_data.begin(), rr_data.end());
//			migraphx::shape r_shape{migraphx::shape::float_type, {num_dirct, hidden_size, hidden_size}};
//			auto r = p.add_literal(migraphx::literal{r_shape, r_data});
//
//			auto bias_data = biasf_data;
//			bias_data.insert(bias_data.end(), biasr_data.begin(), biasr_data.end());
//			migraphx::shape b_shape{migraphx::shape::float_type, {num_dirct, 2 * hidden_size}};
//			auto bias = p.add_literal(migraphx::literal{b_shape, bias_data});
//
//			p.add_instruction(migraphx::op::rnn{hidden_size,
//												{migraphx::op::tanh{}, migraphx::op::tanh{}},
//												migraphx::op::rnn::bidirectional,
//												clip},
//							  seq,
//							  w,
//							  r,
//							  bias,
//							  ih);
//			p.compile(migraphx::cpu::target{});
//
//            //migraphx::program::parameter_map m;
//            //for (auto &&x : p.get_parameter_shapes())
//            //{
//            //    m[x.first] = migraphx::gpu::to_gpu(migraphx::generate_argument(x.second, get_hash(x.first)));
//            //}
//
//			//auto hs_concat = migraphx::gpu::from_gpu(p.eval(m));
//			auto hs_concat = p.eval({});
//			std::vector<float> hs_data;
//			hs_concat.visit([&](auto output) { hs_data.assign(output.begin(), output.end()); });
//
//			std::vector<float> hs_data_gold{
//				0.37780784,  0.61055139,  0.55168478,  -0.5888475, -0.37144644, 0.31708236,
//				0.13104209,  -0.18736027, -0.29385301, 0.16796815, 0.51075965,  0.40258689,
//				-0.13818839, 0.44124447,  0.14365635,  0.14803654, 0.03445704,  0.19167931,
//				-0.3946827,  -0.30889652, -0.22276389, 0.44193283, -0.16477929, -0.11893477,
//				-0.0070999,  0.46251031,  -0.20639211, 0.37488942, -0.0070999,  0.46251031,
//				-0.20639211, 0.37488942};
//
//            for_each(hs_data.begin(), hs_data.end(), [](auto &f) { std::cout << f << "\t"; });
//            std::cout << std::endl;
//		}
//    }
        
    return 0;
}
