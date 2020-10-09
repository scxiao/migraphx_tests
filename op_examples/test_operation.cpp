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

struct simple_operation
{
    template <class T, class F>
    static auto reflect(T& x, F f)
    {
        return migraphx::pack(f(x.data, "data"));
    }
    int data = 1;
    std::string name() const { return "simple"; }
    migraphx::shape compute_shape(const std::vector<migraphx::shape>&) const
    {
        MIGRAPHX_THROW("not computable");
    }
    migraphx::argument compute(migraphx::context&,
                               const migraphx::shape&,
                               const std::vector<migraphx::argument>&) const
    {
        MIGRAPHX_THROW("not computable");
    }
    friend std::ostream& operator<<(std::ostream& os, const simple_operation& op)
    {
        os << op.name() << "[" << op.data << "]";
        return os;
    }
};


int main(int argc, char **argv) {
    migraphx::value v;
    v["a"] = 1;
    v["b"] = 2;

    std::cout << "has_key_a = " << v.contains("a") << std::endl;
    std::cout << "has_key_b = " << v.contains("b") << std::endl;
    std::cout << "v = " << v << std::endl;
    std::cout << "v_size = " << v.size() << std::endl;
    std::cout << "v_a = " << v["a"] << std::endl;
    auto v_a = v[0];
    std::cout << "v_a_key = " << v_a.get_key() << std::endl;
    std::cout << "v_a_val = " << v_a.get_int64() << std::endl;

    migraphx::gpu::context ctx;
    auto ctx_v = ctx.to_value();
    std::cout << "ctx_v = " << ctx_v << std::endl;
    std::cout << "size = " << ctx_v.size() << std::endl;
    std::cout << "has_key = " << ctx_v.contains("events") << std::endl;
    migraphx::gpu::context gtx;
    gtx.from_value(ctx_v);

    return 0;
}

