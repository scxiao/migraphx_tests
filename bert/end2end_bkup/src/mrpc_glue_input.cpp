#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <unordered_map>
#include <migraphx/literal.hpp>
#include <migraphx/operators.hpp>
#include <migraphx/program.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/onnx.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/cpu/target.hpp>
#include <migraphx/gpu/target.hpp>
#include <migraphx/gpu/hip.hpp>

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
    while ((pos_next = sent.find(',', pos)) != std::string::npos)
    {
        auto word_feature = sent.substr(pos, pos_next);
        vec_feature.push_back(std::stoll(word_feature));
        pos = pos_next + 1;
    }
    vec_feature.push_back(std::stoll(sent.substr(pos)));
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
    parse_sentence(line.substr(pos, pos_next), vec_feature);
    pos = pos_next + 1;
    parse_sentence(line.substr(pos), vec_id);
    vec_feature.resize(sent_size, 0);
    vec_id.resize(sent_size, 0);

    // calculate the segment id
    auto it = std::find(vec_id.begin(), vec_id.end(), 1);
    it = std::find(it, vec_id.end(), 0);
    if (it == vec_id.end())
    {
        seg_id.resize(sent_size, 0);
    }
    else
    {
        seg_id.resize(it - vec_id.begin(), 1);
        seg_id.resize(sent_size, 0);
    }

    return label;
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
    if (target_name == "cpu")
    {
        prog.compile(migraphx::cpu::target{});
    }
    else
    {
        prog.compile(migraphx::gpu::target{});
    }

    std::ifstream ifs(argv[2]);
    if (!ifs.is_open())
    {
        std::cout << "Open file " << argv[2] << " error!" << std::endl;
        return 1;
    }

    std::string line;
    std::unordered_map<std::string, std::vector<int64_t>> input_map;
    input_map["input.1"];
    input_map["input.3"];
    input_map["2"];
    migraphx::program::parameter_map m;
    int accu_num = 0, total_num = 0;
    while (std::getline(ifs, line))
    {
        if (line.empty())
        {
            break;
        }

        int label = parse_line(line, 128, input_map);

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
        std::cout << "output = [" << vec_output[0] << ", " << vec_output[1] << "]" << std::endl;
        int calc_label = vec_output[0] >= vec_output[1] ? 0 : 1;
        accu_num += ((calc_label == label) ? 1 : 0);
        ++total_num;
    }

    std::cout << "accuracy rate = " << 1.0 * accu_num / total_num << std::endl;

    return 0;
}

