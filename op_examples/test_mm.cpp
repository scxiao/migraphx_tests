#include "utilities.hpp"

migraphx::program create_program()
{
    migraphx::program p;
    std::vector<float> a = {-0.00925222, 0.56250403, 0.70107397,  0.75402161,  -0.505885,
                        1.33628943,  -0.11413,   -0.31270559, 1.59336732,  -0.19361027,
                        -0.91620867, 0.40108416, -0.06969921, 0.68483471,  -0.39906632,
                        -1.66423624, 0.69040076, -1.31490171, -0.11282616, -0.79391814};
    std::vector<float> b = {6.09568541e-01,
                            -6.10527007e-01,
                            3.66646462e-01,
                            1.18951101e-01,
                            5.58777432e-01,
                            -3.21296298e-01,
                            -5.95997198e-01,
                            -5.01425721e-01,
                            -2.84606807e-01,
                            -5.73673557e-01,
                            -8.99430260e-01,
                            -4.25103093e-01,
                            1.53027987e+00,
                            -3.81407415e-04,
                            -3.29650255e-01};
    std::vector<float> c = {-1.56327541e+00,
                            -7.09570140e-01,
                            -5.37424982e-01,
                            -2.22994831e-01,
                            -2.15586437e+00,
                            2.09177941e-03,
                            -1.47279677e+00,
                            2.02627040e-01,
                            -6.04527691e-01,
                            -1.29885596e+00,
                            2.16294914e+00,
                            -1.48101497e-01};
    migraphx::shape a_shape{migraphx::shape::float_type, {4, 5}};
    auto al = p.add_literal(migraphx::literal{a_shape, a});
    migraphx::shape b_shape{migraphx::shape::float_type, {5, 3}};
    auto bl = p.add_literal(migraphx::literal{b_shape, b});
    migraphx::shape c_shape{migraphx::shape::float_type, {4, 3}};
    auto cl = p.add_literal(migraphx::literal{c_shape, c});
    p.add_instruction(migraphx::op::dot{}, al, bl, cl);

    return p;
}

int main()
{
    auto p1 = create_program();
    std::vector<float> cpu_res, gpu_res;
    run_cpu(p1, cpu_res);
    auto p2 = create_program();
    run_gpu(p2, gpu_res);

    bool b_res = compare_results(cpu_res, gpu_res);
    std::cout << (b_res ? "PASSED!" : "FAILED") << std::endl;

    return 0;
}

