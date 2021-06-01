#ifndef __PARSE_TENSOR_HPP__
#define __PARSE_TENSOR_HPP__

//#include <google/protobuf/text_format.h>
//#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <onnx.pb.h>
#include <string>
#include <migraphx/migraphx.hpp>

//namespace onnx = onnx_for_migraphx;
migraphx::argument parse_tensor(const onnx::TensorProto& t);
migraphx::argument parse_pb_file(const std::string& file_name);

#endif

