#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <migraphx/literal.hpp>
#include <migraphx/operators.hpp>
#include <migraphx/program.hpp>
#include <migraphx/cpu/target.hpp>
#include <migraphx/gpu/target.hpp>
#include <migraphx/gpu/hip.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/onnx.hpp>

migraphx::program load_onnx_file(std::string file_name) {
    auto prog = migraphx::parse_onnx(file_name);

    std::cout << "prog = " << std::endl;
    std::cout << prog << std::endl;

    return prog;
}

void run_cpu(migraphx::program &p)
{
    p.compile(migraphx::cpu::target{});

    migraphx::program::parameter_map m;
    for (auto &&x : p.get_parameter_shapes())
    {
        auto &&argu = migraphx::generate_argument(x.second, get_hash(x.first));
        //m[x.first] = migraphx::gpu::to_gpu(migraphx::generate_argument(x.second, get_hash(x.first)));
        if (x.first == std::string("ih"))
            std::cout << x.first << " = " << argu << std::endl;
        m[x.first] = argu;
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


int main(int argc, char **argv) {
    if (argc < 2) {
        std::cout << "Usage: " << argv[0] << " onnx_file gpu/cpu" << std::endl;
        return 0;
    }

    migraphx::program prog = load_onnx_file(argv[1]);
    prog.compile(migraphx::cpu::target{});

    migraphx::program::parameter_map m;
    std::cout << "Input of rnn is:-----------------------------------" << std::endl;
    for (auto&& x : prog.get_parameter_shapes()) {
        std::cout << x.first << "'s shape:" << std::endl;
        std::cout << x.second << std::endl;
    }
    std::vector<float> res;
    std::cout << "cpu is used." << std::endl;
    for (auto &&x : prog.get_parameter_shapes())
    {
        std::vector<float> data(x.second.elements(), 0.0f);
        if (x.first == "input") {
            data[0] = data[1] = 1.0;
        }
        m[x.first] = migraphx::argument(x.second, &data[0]);
    }

    auto resarg = prog.eval(m);
    resarg.visit([&](auto output) { res.assign(output.begin(), output.end()); } );


    std::cout << "output size = " << res.size() << std::endl;
    for (size_t i = 0; i < res.size(); i++) {
        std::cout << std::setw(12) << res.at(i) <<",";
        if ((i + 1) % 5 == 0) {
            std::cout << std::endl;
        }
    }
    std::cout << std::endl;

    return 0;
}


