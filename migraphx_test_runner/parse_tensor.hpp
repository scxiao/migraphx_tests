#ifndef __PARSE_TENSOR_HPP__
#define __PARSE_TENSOR_HPP__

//#include <google/protobuf/text_format.h>
//#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <onnx.pb.h>
#include <string>
#include <migraphx/migraphx.hpp>

migraphx::argument parse_tensor(const onnx::TensorProto& t, std::vector<std::string>& input_data);
std::vector<char> read_pb_file(const std::string& file_name);
migraphx::argument parse_pb_file(const std::string& file_name, std::vector<std::string>& input_data);

#endif

