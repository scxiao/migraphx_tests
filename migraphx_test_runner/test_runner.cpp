#include <migraphx/migraphx.hpp>
#include "get_cases.hpp"
#include "parse_tensor.hpp"
#include "run_tests.hpp"
#include "cmdline_options.hpp"
#include <string>
#include <unordered_map>

std::unordered_map<std::string, migraphx::argument> wrapup_input_arguments(const std::string& test_case, const std::vector<const char*>& param_names)
{
    std::unordered_map<std::string, migraphx::argument> results;
    std::size_t i = 0;
    for (const auto& name : param_names)
    {
        std::string pb_file_name = test_case + "/input_" + std::to_string(i) + ".pb";
        results[std::string(name)] = parse_pb_file(pb_file_name);
    }

    return results;
}

std::vector<migraphx::argument> get_outputs(const std::string& test_case, const std::size_t out_num)
{
    std::vector<migraphx::argument> results;
    for (std::size_t i = 0; i < out_num; ++i)
    {
        std::string pb_file_name = test_case + "/output_" + std::to_string(i) + ".pb";
        results.push_back(parse_pb_file(pb_file_name));
    }

    return results;
}


migraphx::arguments run_one_case(const std::unordered_map<std::string, migraphx::argument>& inputs, migraphx::program& p)
{
    auto param_shapes = p.get_parameter_shapes();
    migraphx::program_parameters m;
    for (auto&& name : param_shapes.names())
    {
        std::cout << "param_name = " << name << std::endl;
        if (inputs.count(std::string(name)) > 0)
        {
            m.add(name, inputs.at(name));
        }
        else
        {
            auto s = param_shapes[name];
            m.add(name, migraphx::argument::generate(s, 0));
        }
    }

    return p.eval(m);
}


int main(int argc, char **argv)
{
    if (argc < 2)
    {
        std::cout << "Usage: " << argv[0] << " test_loc" << std::endl;
        std::cout << "       -t target: ref/gpu, default: gpu" << std::endl;

        return 0;
    }

    std::string target = "gpu";
    char *target_str = getCmdOption(argv + 2, argv + argc, "-t");
    if (target_str)
    {
        target = std::string(target_str);
    }

    auto model_path_name = get_model_name(argv[1]);
    migraphx::onnx_options parse_options;
    migraphx::program p = migraphx::parse_onnx(model_path_name.c_str(), parse_options);
    auto param_names = p.get_parameter_names();
    auto out_shapes = p.get_output_shapes();
    migraphx_compile_options options;
    options.offload_copy = true;
    p.compile(migraphx::target(target.c_str()), options);

    auto model_name = get_path_last_part(model_path_name);
    auto test_cases = get_test_cases(argv[1]);

    int correct_num = 0;
    for(const auto& test_case : test_cases)
    {
        auto case_name = get_path_last_part(test_case);
        auto inputs = wrapup_input_arguments(test_case, param_names);
        auto outputs = run_one_case(inputs, p);
        auto gold_outputs = get_outputs(test_case, out_shapes.size());

        auto out_num = outputs.size();
        bool correct = true;
        for (std::size_t i = 0; i < out_num; ++i)
        {
            auto gold = gold_outputs.at(i);
            auto output = outputs[i];

            if (gold != output)
            {
                std::cout << "Expected output:" << std::endl;
                std::cout << gold << std::endl;
                std::cout << "..." << std::endl;
                std::cout << "Actual output:" << std::endl;
                std::cout << output << std::endl;
                correct = false;
            }
        }
        std::cout << "Test case: " << test_case << (correct ? "PASSED" : "FAILED") << std::endl;
        correct_num += static_cast<int>(correct);
    }

    std::cout << "Test \"" << get_path_last_part(argv[1]) << "\" has " << test_cases.size() << " cases:" << std::endl;
    std::cout << "\t Passed: " << correct_num << std::endl;
    std::cout << "\t Failed: " << (test_cases.size() - correct_num) << std::endl;

    return 0;
}
