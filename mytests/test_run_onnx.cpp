#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <migraphx/literal.hpp>
#include <migraphx/operators.hpp>
#include <migraphx/program.hpp>
#include <migraphx/quantization.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/gpu/target.hpp>
#include <migraphx/onnx.hpp>
#include "utilities.hpp"
#include "cmd_options.hpp"
#include "read_shape.hpp"

migraphx::program load_onnx_file(std::string file_name, migraphx::onnx_options& options) {
    //auto prog = migraphx::parse_onnx(file_name, options);
    //std::cout << "prog = " << std::endl;
    //std::cout << prog << std::endl;

    std::fstream ifs(file_name.c_str(), std::ios::in | std::ios::binary);
    if (!ifs.is_open())
    {
        std::cout << "File " << file_name << " open error!" << std::endl;
    }
    ifs.seekg(0, ifs.end);
    std::size_t size = ifs.tellg();
    ifs.seekg(0, ifs.beg);
    char *ptr = new char[size];

    ifs.read(ptr, size);

    std::string str(ptr, size);
    auto prog = migraphx::parse_onnx_buffer(str, options);
    std::cout << "prog = " << std::endl;
    std::cout << prog << std::endl;


    //auto tmp_prog = prog;
    //tmp_prog.compile(migraphx::gpu::target{});
    //std::cout << "Compiled_prog = " << std::endl;
    //std::cout << tmp_prog << std::endl;

    return prog;
}

migraphx::target get_target(std::string name)
{
    if (name == "ref")
    {
        return migraphx::ref::target{};
    }
    else
    {
        return migraphx::gpu::target{};
    }
}

void print_args(migraphx::argument& arg, std::ofstream& ofs)
{
    std::vector<float> vec_val;
    arg.visit([&](auto output) { vec_val.assign(output.begin(), output.end()); });

    size_t size = 200;
    for (size_t i = 0; i < size; i++)
    {
        ofs << std::setw(12) << vec_val[i];
        if((i + 1) % 16 == 0)
            ofs << std::endl;
    }
    ofs << std::endl << std::endl;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        std::cout << "Usage: " << argv[0] << " onnx_file" << std::endl;
        std::cout << "\t-d         ref/gpu (default) /both" << std::endl;
        std::cout << "\t-q         fp16/int8/no_quant(default)" << std::endl;
        std::cout << "\t-options   input_shape_file" << std::endl;
        std::cout << "\t-iter      number of runs (default 1)" << std::endl;
        std::cout << "\t-rm        num_ins" << std::endl;
        std::cout << "\t-off_cp    set offload copy" << std::endl; 
        std::cout << "             (default: 0, no offload_copy)" << std::endl;

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

    int num_ins = 0;
    char *num_ins_input = getCmdOption(argv + 2, argv + argc, "-rm");
    if (num_ins_input != nullptr)
    {
        num_ins = std::atoi(num_ins_input);
    }

    int iter_num = 1;
    char *num_iterations = getCmdOption(argv + 2, argv + argc, "-iter");
    if (num_iterations != nullptr)
    {
        iter_num = std::atoi(num_iterations);
    }

    migraphx::onnx_options options;
    char *option_input_file = getCmdOption(argv + 2, argv + argc, "-options");
    if (option_input_file != nullptr)
    {
        options = load_option_file(std::string(option_input_file));
    }

    // whether use offload copy for parameters
    bool offload_copy = false;
    char *use_offload_copy = getCmdOption(argv + 2, argv + argc, "-off_cp");
    if (use_offload_copy != nullptr)
    {
        offload_copy = std::atoi(use_offload_copy);
    }

    auto prog = load_onnx_file(argv[1], options);
    if (num_ins > 0)
    {
        std::cout << "prog_size = " << prog.size() << std::endl;
        std::cout << "Removed the last " << num_ins << " instructions ........." << std::endl;
        auto ins_end = prog.get_main_module()->end();
        auto ins_start = ins_end;
        for (int i = 0; i < num_ins; ++i)
        {
            ins_start = std::prev(ins_start);
        }
        //prog.remove_instructions(ins_start, ins_end);
        prog.get_main_module()->remove_instructions(ins_start, ins_end);

        std::cout << "Program after removing instructions = " << std::endl;
        std::cout << prog << std::endl;
        std::cout << std::endl;
    }

    std::string target_name = "gpu";
    if (device_name == "ref")
    {
        target_name = device_name;
    }
    auto t = get_target(target_name);


    std::cout << "Run on " << device_name << " ........." << std::endl;
    if (quant_flag == "fp16")
    {
        std::cout << "fp16 quantization .............." << std::endl;
        migraphx::quantize_fp16(prog);
    }
    else if (quant_flag == "int8")
    {
        std::cout << "int8 quantization .............." << std::endl;
        auto m = create_param_map(prog);
        std::vector<parameter_map> cali = {m};
        migraphx::quantize_int8(prog, t, cali);
    }
    else
    {
        std::cout << "no quantization ......" << std::endl;
    }

    std::cout << "quant_prog = " << std::endl;
    std::cout << prog << std::endl;
    run_options roptions;
    roptions.offload_copy = offload_copy;
    roptions.t = t;

    if (device_name == "both")
    {
        std::vector<std::vector<float>> ref_res, gpu_res;
        roptions.iter_num = 0;
        roptions.t = migraphx::ref::target{};
        run_prog(prog, ref_res, roptions);
        roptions.iter_num = iter_num;
        roptions.t = migraphx::gpu::target{};
        run_prog(prog, gpu_res, roptions);
        if (ref_res.size() != gpu_res.size())
        {
            std::cout << "CPU and GPU have different number of outputs! " << ref_res.size() << " != " << gpu_res.size() << std::endl;
        }
        std::size_t res_num = ref_res.size();
        bool ret2 = true;
        for (std::size_t i = 0; i < res_num; ++i)
        {
            ret2 = ret2 and compare_results(ref_res[i], gpu_res[i]);
        }
        std::cout << (ret2 ? "PASSED!" : "FAILED") << std::endl;
    }
    else
    {
        std::vector<std::vector<float>> result;
        roptions.iter_num = iter_num;
        run_prog(prog, result, roptions);
        std::cout << "result = " << std::endl;
        print_res(result);
    }

    return 0;
}


