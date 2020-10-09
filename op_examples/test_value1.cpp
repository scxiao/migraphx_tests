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

int main(int argc, char **argv) {
    migraphx::value v2 = {1, 2, 3};
    v2.push_back(2);
    std::cout << "v2_size = " << v2.size() << std::endl;
    v2.visit([](auto v) {
        migraphx::value vv = v;
    });

    migraphx::value v3{"abc", {v2, v2}};
    v3.visit([](auto v) {
        migraphx::value v1 = v;
    });
    std::cout << "v3_is_object = " << v3.is_object() << std::endl;
    std::cout << "v3_object = " << v3.get_object() << std::endl;
    std::cout << "v3 = " << v3 << std::endl;

    return 0;
}

