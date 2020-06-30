
#include "utilities.hpp"

//void run_cpu(migraphx::program &p, std::vector<float> &resData)
//{
//    p.compile(migraphx::cpu::target{});
//
//    migraphx::program::parameter_map m;
//    for (auto &&x : p.get_parameter_shapes())
//    {
//        auto &&argu = migraphx::generate_argument(x.second, get_hash(x.first));
//        m[x.first] = argu;
//        std::cout << "cpu_arg = " << argu << std::endl;
//    }
//    auto result = p.eval(m);
//    result.visit([&](auto output) { resData.assign(output.begin(), output.end()); });
//
//    std::cout << "cpu output_shape = " << result.get_shape() << std::endl;
//    std::cout << "cpu res = " << std::endl;
//    print_res(resData);
//    std::cout << std::endl;
//}
//
//void run_gpu(migraphx::program &p, std::vector<float> &resData)
//{
//    p.compile(migraphx::gpu::target{});
//
//    migraphx::program::parameter_map m;
//    for (auto &&x : p.get_parameter_shapes())
//    {
//        std::cout << "gpu input: " << x.first << "\'shape = " << x.second << std::endl;
//        auto&& argu = migraphx::generate_argument(x.second, get_hash(x.first));
//        std::cout << "gpu_arg = " << argu << std::endl;
//        m[x.first] = migraphx::gpu::to_gpu(argu);
//    }
//    auto result = migraphx::gpu::from_gpu(p.eval(m));
//
//    result.visit([&](auto output) { resData.assign(output.begin(), output.end()); });
//
//
//    std::cout << "gpu output_shape = " << result.get_shape() << std::endl;
//    std::cout << "gpu res = " << std::endl;
//    print_res(resData);
//    std::cout << std::endl;
//}
//
