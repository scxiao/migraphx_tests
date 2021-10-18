#include <string>
#include <vector>
#include <unordered_set>
#include <fstream>
#include <google/protobuf/text_format.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <onnx.pb.h>


static std::vector<std::string> parse_graph(const onnx::GraphProto& graph)
{
    std::unordered_set<std::string> ini_names;
    std::vector<std::string> param_names;
    for (auto&& f : graph.initializer())
    {
        ini_names.insert(f.name());
    }

    for (auto&& input : graph.input())
    {
        const std::string& name = input.name();
        if (not ini_names.count(name) > 0)
        {
            param_names.push_back(name);
        }
    }

    return param_names;
}

std::vector<std::string> get_model_param_names(const std::string& file_name)
{
    std::ifstream input(file_name.c_str(), std::ios::binary);
    if (not input.is_open())
    {
        std::cout << "Error reading model file: " << file_name << std::endl;
        std::abort();
    }

    onnx::ModelProto model;
    if (model.ParseFromIstream(&input))
    {
        if(model.has_graph())
        {
            return parse_graph(model.graph());
        }
    }

    std::cout << "Parse model file \"" << file_name << "\" error!" << std::endl;

    return {};
}


