#include <google/protobuf/text_format.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <onnx.pb.h>
#include <string>
#include <fstream>
#include <iostream>
#include <unordered_map>

using node_map = std::unordered_map<std::string, onnx::NodeProto>;

static node_map get_nodes(const onnx::GraphProto& graph)
{
    std::unordered_map<std::string, onnx::NodeProto> result;
    std::size_t n = 0;
    for(auto&& node : graph.node())
    {
        if(node.output().empty())
        {
            if(node.name().empty())
            {
                result["migraphx_unamed_node_" + std::to_string(n)] = node;
                n++;
            }
            else
            {
                result[node.name()] = node;
            }
        }
        for(auto&& output : node.output())
        {
            result[output] = node;
        }
    }
    return result;
}


void parse_graph(const onnx::GraphProto& graph)
{
	node_map nodes = get_nodes(graph);    
}

void parse_onnx(const std::string& file_name)
{
    std::fstream input(file_name.c_str(), std::ios::in | std::ios::binary);
	if (!input.is_open())
    {
        std::cout << "File " << file_name << " open error!" << std::endl;
        return;
    }

    onnx::ModelProto model;
    if (model.ParseFromIstream(&input))
    {
        if (model.has_graph())
        {
            parse_graph(model.graph());
        }
    }
}

int main(int argc, char **argv)
{
    if (argc != 2)
    {
        std::cout << "Usage: " << argv[0] << " onnx_file" << std::endl;
        return 0;
    }

    parse_onnx(std::string(argv[1]));

    return 0;
}

