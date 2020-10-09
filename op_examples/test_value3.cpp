#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <migraphx/literal.hpp>
#include <migraphx/quantization.hpp>
#include <migraphx/operators.hpp>
#include <migraphx/program.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/gpu/target.hpp>
#include <migraphx/cpu/target.hpp>
#include <migraphx/value.hpp>
#include <migraphx/serialize.hpp>
#include <migraphx/json.hpp>

int main(int argc, char **argv) {
//    migraphx::value v = {1, 2};
//    migraphx::value v0 = 1;
//    std::string str;
//    migraphx::value_to_json_string(v0, str);
//    std::cout << "str = " << str << std::endl;

    std::string str = "[]";
    migraphx::value e = migraphx::from_json_string(str);
    std::cout << "is_array = " << e.is_array() << std::endl;
    std::cout << "is_object = " << e.is_object() << std::endl;
    std::cout << "is_empty = " << e.empty() << std::endl;
    std::cout << "e = " << e << std::endl;
    migraphx::value ev{};
    std::cout << "ev = " << ev << std::endl;

//    v0["a"] = 1;
//    v0["b"] = 2;
//    std::string str0;
//    migraphx::value_to_json_string(v0, str0);
//    std::cout << "v0 = " << v0 << std::endl;
//    std::cout << "str0 = " << str0 << std::endl;
//
//    migraphx::value v1;
//    migraphx::value_from_json_string(str0, v1);
//    std::cout << "v1 = " << v1 << std::endl;
//
//    std::string str1;
//    migraphx::value_to_json_string(v1, str1);
//    std::cout << "str1 = " << str1 << std::endl;
//
//    migraphx::value v2;
//    migraphx::value_from_json_string(str1, v2);
//    std::cout << "v2 = " << v2 << std::endl;
//
//    std::string str2;
//    migraphx::value_to_json_string(v2, str2);
//    std::cout << "str2 = " << str2 << std::endl;


    return 0;
}

