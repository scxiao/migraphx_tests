#ifndef __READ_SHAPE_HPP__
#define __READ_SHAPE_HPP__

#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <migraphx/literal.hpp>
#include <migraphx/operators.hpp>
#include <migraphx/program.hpp>
#include <migraphx/quantization.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/gpu/target.hpp>
#include <migraphx/onnx.hpp>
#include "cmd_options.hpp"
#include "read_shape.hpp"


std::string process_one_line(std::string line, std::vector<std::size_t>& dims);
migraphx::onnx_options load_option_file(std::string file);

#endif
