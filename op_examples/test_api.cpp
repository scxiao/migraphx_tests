#include <iostream>
#include <migraphx/migraphx.h>
#include <migraphx/migraphx.hpp>

int main(int argc, char **argv) {
    if (argc != 2)
    {
        std::cout << "Usage: " << argv[0] << " onnx_file" << std::endl;
        return 0;
    }

    migraphx::program prog = migraphx::parse_onnx(argv[1]);
    std::cout << "prog = " << std::endl;
    prog.print();
    std::cout << std::endl;

    migraphx::quantize_fp16(prog);
    std::cout << "quantized_prog = " << std::endl;
    prog.print();
    std::cout << std::endl;

    prog.compile(migraphx::target("gpu"));
    std::cout << "compiled_prog = " << std::endl;
    prog.print();
    std::cout << std::endl;

    return 0;
}
