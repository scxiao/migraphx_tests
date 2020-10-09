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
#include <migraphx/serialize.hpp>
#include <migraphx/json.hpp>

int main(int argc, char **argv) {
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
    std::cout << "ll_str0 = " << l_str << std::endl;

    migraphx::value v_json = migraphx::from_json_string(l_str);
    std::cout << "v_json = " << v_json << std::endl;

    std::string ll_str = migraphx::to_json_string(v_json);
    std::cout << "ll_str1 = " << ll_str << std::endl;

    migraphx::value v_json1 = migraphx::from_json_string(ll_str);
    std::cout << "v_json1 = " << v_json1 << std::endl;

    std::string ll_str2 = migraphx::to_json_string(v_json1);
    std::cout << "ll_str2 = " << ll_str2 << std::endl;

    migraphx::value v_json2 = migraphx::from_json_string(ll_str2);
    std::cout << "v_json2 = " << v_json2 << std::endl;



    return 0;
}

