#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <migraphx/migraphx.hpp>
#include "cmdline_options.hpp"
#include "load_shape_file.hpp"
#include "utilities.hpp"

migraphx::program load_onnx_file(std::string file_name, migraphx::onnx_options options) {
    auto prog = migraphx::parse_onnx(file_name.c_str(), options);
    std::cout << "Load program is: " << std::endl;
    prog.print();
    std::cout << std::endl;

    return prog;
}

migraphx::target get_target(std::string name)
{
    return migraphx::target(name.c_str());
}


int main(int argc, char **argv) {
    if (argc < 2) {
        std::cout << "Usage: " << argv[0] << " onnx_file [options]" << std::endl;
        std::cout << "options:" << std::endl;
        std::cout << "\t-s       shape_info_file" << std::endl;
        std::cout << "\t-d       ref/gpu(default)/both" << std::endl;
        std::cout << "\t-q       fp16/int8/no_quant(default)" << std::endl;
        std::cout << "\t-iter    iter_num (default: 1)" << std::endl;
        return 0;
    }

    std::string device_name("gpu");
    char* dev_name = getCmdOption(argv + 2, argv + argc, "-d");
    if (dev_name != nullptr)
    {
        std::string dev_str(dev_name);
        if (dev_str == "ref" or dev_str == "both")
        {
            device_name = dev_str;
        }
    }

    std::string quant_flag;
    char *quant_input = getCmdOption(argv + 2, argv + argc, "-q");
    if (quant_input != nullptr)
    {
        quant_flag.append(quant_input);
    }

    // process the shape info
    migraphx::onnx_options options;
    char *option_input_file = getCmdOption(argv + 2, argv + argc, "-s");
    if (option_input_file != nullptr)
    {
        options = load_name_dim_file(std::string(option_input_file));
    }

    std::size_t iter_num = 1;
    char *option_iter_num = getCmdOption(argv + 2, argv + argc, "-iter");
    if (option_iter_num != nullptr)
    {
        iter_num = std::atoi(option_iter_num);
    }

    migraphx::program prog = load_onnx_file(argv[1], options);
    std::string target_name = "gpu";
    if (device_name == "ref")
    {
        target_name = device_name;
    }
    auto t = get_target(target_name);

    std::cout << "Run on " << device_name << " ........." << std::endl;
    //quantize the program
    if (quant_flag == "fp16")
    {
        std::cout << "fp16 quantization ................" << std::endl;
        migraphx::quantize_fp16(prog);
        std::cout << "quant_prog = " << std::endl;
        prog.print();
        std::cout << std::endl;
    }
    else if (quant_flag == "int8")
    {
        std::cout << "int8 quantization ................" << std::endl;
        std::cout << "quant_prog = " << std::endl;
        prog.print();
        std::cout << std::endl;
    }
    else
    {
        std::cout << "No quantization .................." << std::endl;
    }

    if (device_name == "both")
    {
        migraphx::program prog_g = migraphx::parse_onnx(argv[1], options);
        std::vector<std::vector<float>> cpu_res, gpu_res;
        run_prog(prog, migraphx::target("ref"), cpu_res);
        run_prog(prog_g, migraphx::target("gpu"), gpu_res);
        if (cpu_res.size() != gpu_res.size())
        {
            std::cout << "CPU and GPU have different number of outputs! " << cpu_res.size() << " != " << gpu_res.size() << std::endl;
        }
        std::size_t res_num = cpu_res.size();
        bool ret2 = true;
        for (std::size_t i = 0; i < res_num; ++i)
        {
            ret2 = ret2 and compare_results(cpu_res[i], gpu_res[i]);
        }
        std::cout << (ret2 ? "PASSED!" : "FAILED") << std::endl;
    }
    else
    {
        std::vector<std::vector<float>> result;
        run_prog(prog, t, result, iter_num);
        //std::cout << "result = " << std::endl;
        //print_res(result);
    }

    return 0;
}

