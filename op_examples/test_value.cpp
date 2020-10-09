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
//    migraphx::value v1 = 1;
//    std::cout << "v1_size = " << v1.size() << std::endl;
//    v1.visit([](auto v) {
//            migraphx::value vv = v;
//    });
//    int64_t in = v1.get_int64();
//    std::cout << "in = " << in << std::endl;
//
//    migraphx::value v2 = {1, 2, 3};
//    v2.push_back(2);
//    std::cout << "v2_size = " << v2.size() << std::endl;
//    v2.visit([](auto v) {
//        migraphx::value vv = v;
//    });
//
//    migraphx::value v3{"abc", {v2, v2}};
//    v3.visit([](auto v) {
//        migraphx::value v1 = v;
//    });
//    std::cout << "v3_size = " << v3.size() << std::endl;
//    std::cout << "v3_key = " << v3.get_key() << std::endl;
//    std::cout << "v3_is_object = " << v3.is_object() << std::endl;
//    std::cout << "v3_object = " << v3.get_object() << std::endl;
//    std::cout << "v3 = " << v3 << std::endl;
//
    migraphx::shape s{migraphx::shape::float_type, {2, 3}};
    migraphx::value s_val = migraphx::to_value(s);
    std::cout << "s_val = " << s_val << std::endl;

    std::vector<float> vec(s.elements());
    std::iota(vec.begin(), vec.end(), 1);
    migraphx::argument argu = migraphx::argument(s, vec.data());
    migraphx::value v_argu;
    migraphx::migraphx_to_value(v_argu, argu);
    std::cout << "argu_val = " << v_argu << std::endl;
    std::cout << "argu_val_size = " << v_argu.size() << std::endl;

    std::string argu_str = migraphx::to_json_string(v_argu);
    std::cout << "json_argu = " << argu_str << std::endl;

    migraphx::literal l = migraphx::literal(migraphx::literal(s, vec));
    migraphx::value v_l;
    migraphx::migraphx_to_value(v_l, l);

    std::cout << "literal_val = " << v_l << std::endl;
    std::cout << "literal_val_size = " << v_l.size() << std::endl;

    std::string l_str = migraphx::to_json_string(v_l);
    std::cout << "json_literal = " << l_str << std::endl;

    migraphx::value v_json = migraphx::from_json_string(l_str);

    migraphx::value tmp_v1;
    migraphx::value tmp_v2{};
    migraphx::value tmp_v3 = migraphx::value();
    migraphx::value tmp_v4 = migraphx::value{};

    return 0;
}

