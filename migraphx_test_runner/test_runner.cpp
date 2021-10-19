#include <migraphx/migraphx.hpp>
#include "get_cases.hpp"
#include "parse_tensor.hpp"
#include "run_tests.hpp"
#include "cmdline_options.hpp"
#include "get_model_param_names.hpp"
#include <string>
#include <unordered_map>

static std::unordered_map<std::string, migraphx::argument> get_input_from_files(const std::string& test_case,
        std::vector<std::string> param_names,
        std::vector<std::string>& input_data)
{
    std::unordered_map<std::string, migraphx::argument> results;
    std::size_t i = 0;
    for (auto name : param_names)
    {
        std::string pb_file_name = test_case + "/input_" + std::to_string(i++) + ".pb";
        results[name] = parse_pb_file(pb_file_name, input_data);
    }

    return results;
}

static std::unordered_map<std::string, std::vector<std::size_t>> get_input_shapes(const std::string& test_case, std::vector<std::string> param_names)
{
    std::unordered_map<std::string, std::vector<std::size_t>> results;
    std::size_t i = 0;
    for (auto name : param_names)
    {
        std::string pb_file_name = test_case + "/input_" + std::to_string(i++) + ".pb";
        results[name] = get_input_dims(pb_file_name);
    }

    return results;
}

static std::vector<migraphx::argument> get_outputs(const std::string& test_case, const std::size_t out_num, std::vector<std::string>& out_data)
{
    std::vector<migraphx::argument> results;
    for (std::size_t i = 0; i < out_num; ++i)
    {
        std::string pb_file_name = test_case + "/output_" + std::to_string(i) + ".pb";
        results.push_back(parse_pb_file(pb_file_name, out_data));
    }

    return results;
}

static migraphx::arguments run_one_case(const std::unordered_map<std::string, migraphx::argument>& inputs, migraphx::program& p)
{
    auto param_shapes = p.get_parameter_shapes();
    migraphx::program_parameters m;
    for (auto&& name : param_shapes.names())
    {
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

static bool tune_param_shape(const migraphx::program& p, const std::unordered_map<std::string, migraphx::argument>& inputs, 
                                                migraphx::onnx_options& options)
{
    bool ret = false;
    auto param_shapes = p.get_parameter_shapes();
    for(const auto& name : param_shapes.names())
    {
        std::string nm(name);
        if (inputs.count(nm) > 0)
        {
            auto param_s = param_shapes[name];
            auto data_s = inputs.at(nm).get_shape();
            if (not compare_shapes(param_s, data_s))
            {
                options.set_input_parameter_shape(nm, data_s.lengths());
                ret = true;
            }
        }
    }

    return ret;
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

    std::cout << "Run test \"" << argv[1] << "\" on \"" << target << "\":" << std::endl << std::endl;

    auto model_path_name = get_model_name(argv[1]);
    auto param_names = get_model_param_names(model_path_name);


    // retrieve all test cases
    auto test_cases = get_test_cases(argv[1]);
    // sample test case
    auto sample_test_case = test_cases.front();
    auto param_shapes = get_input_shapes(sample_test_case, param_names);

    std::cout << "Model parameter name and shapes:" << std::endl;
    for(const auto& name : param_names)
    {
        std::cout << "name: " << name << ", shape = " << param_shapes[name] << std::endl;
    }
    std::cout << std::endl;

    // set the param to corresponding dims
    migraphx::onnx_options parse_options;
    for(const auto& name : param_names)
    {
        if (param_shapes.count(name) > 0)
        {
            auto dims = param_shapes[name];
            parse_options.set_input_parameter_shape(name, dims);
        }
    }

    std::cout << "Parsing input model \"" << model_path_name << "\" ......" << std::endl;
    migraphx::program p = migraphx::parse_onnx(model_path_name.c_str(), parse_options);
    std::cout << "Done parsing model ......" << std::endl;

    auto out_shapes = p.get_output_shapes();
    migraphx::compile_options options;
    options.set_offload_copy();

    std::cout << "\nCompiling program ..." << std::endl;
    p.compile(migraphx::target(target.c_str()), options);
    std::cout << "Done compiling program ..." << std::endl;

    auto model_name = get_path_last_folder(model_path_name);

    std::cout << "\nBegin execution ..... " << std::endl;
    int correct_num = 0;
    for(const auto& test_case : test_cases)
    {
        auto case_name = get_path_last_folder(test_case);
        std::vector<std::string> input_data;
        auto inputs = get_input_from_files(test_case, param_names, input_data);
        migraphx::onnx_options parse_options;
        if (tune_param_shape(p, inputs, parse_options))
        {
            p = migraphx::parse_onnx(model_path_name.c_str(), parse_options);
            p.compile(migraphx::target(target.c_str()), options);
        }

        auto outputs = run_one_case(inputs, p);
        std::vector<std::string> out_data;
        auto gold_outputs = get_outputs(test_case, out_shapes.size(), out_data);

        auto out_num = outputs.size();
        bool correct = true;
        for (std::size_t i = 0; i < out_num; ++i)
        {
            auto gold = gold_outputs.at(i);
            auto output = outputs[i];

            if (not compare_results(gold, output))
            {
                std::cout << "Expected output:" << std::endl;
                std::cout << gold << std::endl;
                std::cout << "..." << std::endl;
                std::cout << "Actual output:" << std::endl;
                std::cout << output << std::endl;
                correct = false;
            }
        }
        std::cout << "\tTest case \"" << case_name << "\": " << (correct ? "PASSED" : "FAILED") << std::endl;
        correct_num += static_cast<int>(correct);
    }

    std::cout << "\nTest \"" << argv[1] << "\" has " << test_cases.size() << " cases:" << std::endl;
    std::cout << "\t Passed: " << correct_num << std::endl;
    std::cout << "\t Failed: " << (test_cases.size() - correct_num) << std::endl;

    return 0;
}
