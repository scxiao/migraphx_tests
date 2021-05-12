#include <google/protobuf/text_format.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <onnx.pb.h>
#include <string>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <unordered_map>

using node_map = std::unordered_map<std::string, onnx::NodeProto>;
void print_indent(int n)
{
    for (int m = 0; m < n; m++)
        std::cout << "        ";
}

void parse_graph(const onnx::GraphProto& graph, int n)
{
    std::cout << "Initializer names = " << std::endl;
    for(auto&& f : graph.initializer())
    {
        print_indent(n);
        std::cout << "\t" << f.name() << std::endl;
    }

    std::cout << "Input names = " << std::endl;
    for(auto&& input : graph.input())
    {
        const std::string& name = input.name();
        print_indent(n);
        std::cout << "\t " << name << std::endl;
    }

    print_indent(n);
    std::cout << "Node_info = " << std::endl;
    int i = 0;
    for(auto&& node : graph.node())
    {
        print_indent(n);
        std::cout << std::setw(8) << i++;
        std::cout << "\t" << node.op_type() << std::endl;
        if (node.op_type() == "Loop")
        {
			for(auto&& attr : node.attribute())
			{
                auto&& sub_graph = attr.g();
                print_indent(n);
                std::cout << "n = " << n << ", sub_graph = " << std::endl;
                parse_graph(sub_graph, n + 1);
			}
        }
    }

    std::cout << std::endl;
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
            parse_graph(model.graph(), 1);
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

