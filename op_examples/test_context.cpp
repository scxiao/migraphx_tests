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
#include <migraphx/value.hpp>
#include <migraphx/serialize.hpp>
#include <migraphx/json.hpp>
#include <migraphx/gpu/context.hpp>
#include <migraphx/context.hpp>

int main(int argc, char **argv) {
    migraphx::context ctx = migraphx::gpu::context{};
    auto ctx_v = ctx.to_value();
    std::cout << "ctx_v = " << ctx_v << std::endl;
    std::cout << "size = " << ctx_v.size() << std::endl;
    std::cout << "has_key = " << ctx_v.contains("events") << std::endl;
    auto v_events = ctx_v.at("events");
    std::cout << "v_events = " << v_events << std::endl;
    auto n_events = v_events.to<std::pair<std::string, std::size_t>>();


    migraphx::gpu::context gtx;
    gtx.from_value(ctx_v);

    return 0;
}

