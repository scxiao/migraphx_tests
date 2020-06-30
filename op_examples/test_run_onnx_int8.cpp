#include <iostream>
#include <vector>
#include <string>
#include <migraphx/literal.hpp>
#include <migraphx/quantization.hpp>
#include <migraphx/operators.hpp>
#include <migraphx/program.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/gpu/target.hpp>
#include <migraphx/onnx.hpp>
#include "utilities.hpp"

migraphx::program load_onnx_file(std::string file_name) {
    auto prog = migraphx::parse_onnx(file_name);
    return prog;
}

int main(int argc, char **argv) {
    if (argc != 2) {
        std::cout << "Usage: " << argv[0] << " onnx_file" << std::endl;
        return 0;
    }

//    auto print_conv_inputs = [&](std::size_t ins_index, std::vector<migraphx::argument> args)
//    {
//        std::ofstream ofs("quant_conv_res.txt", std::ios::app);
//        std::vector<float> vec_val;
//        args.front().visit([&](auto output) { vec_val.assign(output.begin(), output.end()); });
//        auto max_abs = std::fabs(vec_val[0]);
//        for (size_t i = 1; i < vec_val.size(); i++)
//        {
//            max_abs = max_abs > std::fabs(vec_val[i]) ? max_abs : std::fabs(vec_val[i]);
//        }
//        ofs << "run_onnx, before_quant8, ins_index = " << ins_index << ", max_abs = " << max_abs << std::endl;
//        ofs << "run_onnx, before_quant8, capture_input = " << std::endl;
//        size_t size = 200;
//        for (size_t i = 0; i < size; i++)
//        {
//            ofs << std::setw(12) << vec_val[i];
//            if((i + 1) % 16 == 0)
//                ofs << std::endl;
//        }
//        ofs << std::endl << std::endl;
//        ofs.close();
//    };

    auto cap_prog = load_onnx_file(argv[1]);
    auto prog = cap_prog;

    // add capture operator for each operator in the op_names
    std::vector<std::string> op_names;
    op_names.push_back("convolution");
    op_names.push_back("dot");
    migraphx::capture_arguments(cap_prog, op_names);

    std::cout << "captured_prog = " << cap_prog << std::endl;

    // compile the program and run it once to get the scale and
    // shift for each convert operator
    cap_prog.compile(migraphx::cpu::target{});
    std::vector<float> cap_res;
    run_cpu(cap_prog, cap_res);

    // after running the model, we have the scale and shift for each convert 
    // operator to quantize a program to the int8 type
    // migraphx::capture_arguments(prog, op_names, print_conv_inputs);
    migraphx::quantize_int8(prog, op_names);
    //migraphx::quantize(prog);
    std::cout << "quant_prog = " << prog << std::endl;

    std::vector<float> cpu_res, gpu_res;
    run_cpu(prog, cpu_res);
    run_gpu(prog, gpu_res);
    bool ret1 = compare_results(cpu_res, cap_res);
    std::cout << (ret1 ? "PASSED!" : "FAILED") << std::endl;
    bool ret2 = compare_results(cpu_res, gpu_res);
    std::cout << (ret2 ? "PASSED!" : "FAILED") << std::endl;

    return 0;
}

