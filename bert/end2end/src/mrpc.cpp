#include <iostream>
#include <vector>
#include <ctime>
#include <unordered_map>
#include <string>
#include <fstream>
#include <migraphx/literal.hpp>
#include <migraphx/operators.hpp>
#include <migraphx/program.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/quantization.hpp>
#include <migraphx/onnx.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/cpu/target.hpp>
#include <migraphx/gpu/target.hpp>
#include <migraphx/gpu/hip.hpp>

using namespace std::chrono;

migraphx::program load_onnx_file(std::string file_name) {
    auto prog = migraphx::parse_onnx(file_name);
    std::cout << "prog = " << std::endl;
    std::cout << prog << std::endl;
    return prog;
}

template <class T>
auto get_hash(const T& x)
{
    return std::hash<T>{}(x);
}

void parse_sentence(const std::string& sent, std::vector<int64_t>& vec_feature)
{
    size_t pos = 0, pos_next;
    size_t index = 0;
    while ((pos_next = sent.find(',', pos)) != std::string::npos)
    {
        auto word_feature = sent.substr(pos, pos_next);
        vec_feature.push_back(std::stoll(word_feature));
        pos = pos_next + 1;
    }
    vec_feature.push_back(std::stoll(sent.substr(pos)));
    vec_feature.push_back(102);
}

int parse_line(std::string& line, std::size_t sent_size, 
        std::unordered_map<std::string, std::vector<int64_t>>& input_map)
{
    auto& vec_feature = input_map["input.1"];
    auto& vec_id = input_map["input.3"];
    auto& seg_id = input_map["2"];
    vec_feature.clear();
    vec_id.clear();
    seg_id.clear();

    size_t pos = line.find('\t');
    int label = std::stoi(line.substr(0, pos));

    ++pos;
    size_t pos_next = line.find('\t', pos);
    vec_feature.push_back(101);
    parse_sentence(line.substr(pos, pos_next), vec_feature);
    vec_id.resize(vec_feature.size(), 0);

    pos = pos_next + 1;
    parse_sentence(line.substr(pos), vec_feature);
    vec_id.resize(vec_feature.size(), 1);
    seg_id.resize(vec_feature.size(), 1);

    vec_feature.resize(sent_size, 0);
    vec_id.resize(sent_size, 0);
    seg_id.resize(sent_size, 0);

    return label;
}

void print_vec(std::vector<int64_t>& vec)
{
    for (auto v : vec)
    {
        std::cout << v << ", ";
    }
    std::cout << std::endl;
}

int main(int argc, char **argv) {
    if (argc != 4) {
        std::cout << "Usage: " << argv[0] << " onnx_file input_data cpu/gpu" << std::endl;
        return 0;
    }

    std::string target_name("cpu");
    if (std::string(argv[3]) == std::string("gpu"))
    {
        target_name = "gpu";
    }

    auto prog = load_onnx_file(argv[1]);
    //migraphx::quantize(prog);
    //std::cout << "quantized prog = " << std::endl;
    std::cout << prog << std::endl;
    if (target_name == "cpu")
    {
        prog.compile(migraphx::cpu::target{});
    }
    else
    {
        prog.compile(migraphx::gpu::target{});
    }

    std::size_t batch_size = 1;
    auto param_shapes = prog.get_parameter_shapes();
    if (param_shapes.count("input.1") > 0)
    {
        batch_size = prog.get_parameter_shapes()["input.1"].lens()[0];
    }

    std::ifstream ifs(argv[2]);
    if (!ifs.is_open())
    {
        std::cout << "Open file " << argv[2] << " error!" << std::endl;
        return 1;
    }

    std::string line;
    std::getline(ifs, line);
    migraphx::program::parameter_map m;
    std::size_t accu_count = 0, total_count = 0;
    high_resolution_clock::time_point t1 = high_resolution_clock::now();
    while (true)
    {
        std::unordered_map<std::string, std::vector<int64_t>> input_map;
        input_map["input.1"];
        input_map["input.3"];
        input_map["2"];
        std::unordered_map<std::string, std::vector<int64_t>> sent_tokens;
        sent_tokens["input.1"];
        sent_tokens["input.3"];
        sent_tokens["2"];
        std::vector<int> vec_labels;

        for (std::size_t batch_no = 0; batch_no < batch_size; batch_no++)
        {
            std::getline(ifs, line);
            if (line.empty())
            {
                break;
            }
            int label = parse_line(line, 128, sent_tokens);
            vec_labels.push_back(label);
            input_map["input.1"].insert(input_map["input.1"].end(), sent_tokens["input.1"].begin(),
                    sent_tokens["input.1"].end());
            input_map["input.3"].insert(input_map["input.3"].end(), sent_tokens["input.3"].begin(),
                    sent_tokens["input.3"].end());
            input_map["2"].insert(input_map["2"].end(), sent_tokens["2"].begin(),
                    sent_tokens["2"].end());
        }
       
        if (line.empty())
        {
            break;
        }

        for (auto &&x : prog.get_parameter_shapes())
        {
            //std::cout << "gpu input: " << x.first << ", shape = " << x.second << std::endl;
            migraphx::argument argu{};
            if (input_map.count(x.first) > 0)
            {
                argu = migraphx::argument(x.second, input_map[x.first].data());
            }
            else
            {
                argu = migraphx::generate_argument(x.second, get_hash(x.first));
            }

            if (target_name == "cpu")
            {
                m[x.first] = argu;
            }
            else
            {
                m[x.first] = migraphx::gpu::to_gpu(argu);
            }
        }

        migraphx::argument result{};
        if (target_name == "cpu")
        {
            result = prog.eval(m);
        }
        else
        {
            result = migraphx::gpu::from_gpu(prog.eval(m));
        }
        std::vector<float> vec_output;
        result.visit([&](auto output) { vec_output.assign(output.begin(), output.end()); });

        for (std::size_t batch_no = 0; batch_no < batch_size; ++batch_no)
        {
            std::cout << "[" << vec_output[2 * batch_no] << ", " << vec_output[2 * batch_no + 1] << "]" << std::endl;
            int calc_label = (vec_output[2 * batch_no] >= vec_output[2 * batch_no + 1]) ? 0 : 1;
            accu_count += (calc_label == vec_labels[batch_no]) ? 1 : 0;
            ++total_count;
        }
    }

    std::cout << "accuracy rate = " << 1.0 * accu_count / total_count << std::endl;

    high_resolution_clock::time_point t2 = high_resolution_clock::now();
    
    milliseconds ms = duration_cast<milliseconds>(t2 - t1);
    std::cout << "It takes " << ms.count() << " ms" << std::endl;
    return 0;
}

