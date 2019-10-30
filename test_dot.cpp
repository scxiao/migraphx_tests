#include "utilities.hpp"

// cases with 3 inputs
migraphx::program create_program_vv1(float alpha = 1.0, float beta = 1.0f)
{
    migraphx::program p;
    std::vector<float> a = {0.7481789 ,  0.02906279,  1.01193836,  1.60222907,  1.89135978,
        0.30054158, -0.4892588 , -0.27027533};
    std::vector<float> b = {-0.25829116,  0.27908929, -1.27888957,  0.21152361,  0.08593658,
        0.52163899,  1.38343824, -0.2342857};
    migraphx::shape a_shape{migraphx::shape::float_type, {8}};
    auto al = p.add_literal(migraphx::literal{a_shape, a});
    auto ual = p.add_instruction(migraphx::op::unsqueeze{{0}}, al);
    migraphx::shape b_shape{migraphx::shape::float_type, {8}};
    auto bl = p.add_literal(migraphx::literal{b_shape, b});
    auto ubl = p.add_instruction(migraphx::op::unsqueeze{{1}}, bl);
    p.add_instruction(migraphx::op::dot{alpha}, ual, ubl);

    return p;
}

migraphx::program create_program_mm_c1(float alpha = 1.0, float beta = 1.0f)
{
    migraphx::program p;
    std::vector<float> a = {-0.86217194, -1.04129542, -0.64850364, -0.97078327,
       -0.40516386,  0.83136927,  0.37717502,  0.42271939,
        1.10062165, -0.92239359,  0.40403076, -0.43935377};
    std::vector<float> b = {0.76084386,  1.89201125,  1.73218067,
        0.7148568 , -0.55578914,  0.05799101,
       -1.24090721, -0.51151978,  1.13255803,
        0.21540723, -1.10459009,  0.45580331};
	std::vector<float> c = {-0.80473623,  0.35154171, -2.73077756,
       -0.09093885, -1.88850472, -0.03375556,
       -0.41798276,  2.87368099,  2.11031439};

    migraphx::shape a_shape{migraphx::shape::float_type, {3, 4}};
    auto al = p.add_literal(migraphx::literal{a_shape, a});
    migraphx::shape b_shape{migraphx::shape::float_type, {4, 3}};
    auto bl = p.add_literal(migraphx::literal{b_shape, b});
    migraphx::shape c_shape{migraphx::shape::float_type};
    auto cl = p.add_literal(migraphx::literal{c_shape, {1}});
    p.add_instruction(migraphx::op::dot{alpha, beta}, al, bl, cl);

    return p;
}

migraphx::program create_program_mm_c3(float alpha = 1.0, float beta = 1.0f)
{
    migraphx::program p;
    std::vector<float> a = {-0.86217194, -1.04129542, -0.64850364, -0.97078327,
       -0.40516386,  0.83136927,  0.37717502,  0.42271939,
        1.10062165, -0.92239359,  0.40403076, -0.43935377};
    std::vector<float> b = {0.76084386,  1.89201125,  1.73218067,
        0.7148568 , -0.55578914,  0.05799101,
       -1.24090721, -0.51151978,  1.13255803,
        0.21540723, -1.10459009,  0.45580331};
	std::vector<float> c = {-0.80473623,  0.35154171, -2.73077756,
       -0.09093885, -1.88850472, -0.03375556,
       -0.41798276,  2.87368099,  2.11031439};

    migraphx::shape a_shape{migraphx::shape::float_type, {3, 4}};
    auto al = p.add_literal(migraphx::literal{a_shape, a});
    migraphx::shape b_shape{migraphx::shape::float_type, {4, 3}};
    auto bl = p.add_literal(migraphx::literal{b_shape, b});
    migraphx::shape c_shape{migraphx::shape::float_type, {3, 1}};
    std::vector<float> vec_c(3, 2.0f);
    auto cl = p.add_literal(migraphx::literal{c_shape, vec_c});
    p.add_instruction(migraphx::op::dot{alpha, beta}, al, bl, cl);

    return p;
}

migraphx::program create_program_mm_c31(float alpha = 1.0, float beta = 1.0f)
{
    migraphx::program p;
    std::vector<float> a = {-0.86217194, -1.04129542, -0.64850364, -0.97078327,
       -0.40516386,  0.83136927,  0.37717502,  0.42271939,
        1.10062165, -0.92239359,  0.40403076, -0.43935377};
    std::vector<float> b = {0.76084386,  1.89201125,  1.73218067,
        0.7148568 , -0.55578914,  0.05799101,
       -1.24090721, -0.51151978,  1.13255803,
        0.21540723, -1.10459009,  0.45580331};
	std::vector<float> c = {-0.80473623,  0.35154171, -2.73077756,
       -0.09093885, -1.88850472, -0.03375556,
       -0.41798276,  2.87368099,  2.11031439};

    migraphx::shape a_shape{migraphx::shape::float_type, {3, 4}};
    auto al = p.add_literal(migraphx::literal{a_shape, a});
    migraphx::shape b_shape{migraphx::shape::float_type, {4, 3}};
    auto bl = p.add_literal(migraphx::literal{b_shape, b});
    migraphx::shape c_shape{migraphx::shape::float_type, {3}};
    std::vector<float> vec_c(3, 2.0f);
    auto cl = p.add_literal(migraphx::literal{c_shape, vec_c});
    p.add_instruction(migraphx::op::dot{alpha, beta}, al, bl, cl);

    return p;
}

migraphx::program create_program_mm_c33(float alpha = 1.0, float beta = 1.0f)
{
    migraphx::program p;
    std::vector<float> a = {-0.86217194, -1.04129542, -0.64850364, -0.97078327,
       -0.40516386,  0.83136927,  0.37717502,  0.42271939,
        1.10062165, -0.92239359,  0.40403076, -0.43935377};
    std::vector<float> b = {0.76084386,  1.89201125,  1.73218067,
        0.7148568 , -0.55578914,  0.05799101,
       -1.24090721, -0.51151978,  1.13255803,
        0.21540723, -1.10459009,  0.45580331};
	std::vector<float> c = {-0.80473623,  0.35154171, -2.73077756,
       -0.09093885, -1.88850472, -0.03375556,
       -0.41798276,  2.87368099,  2.11031439};

    migraphx::shape a_shape{migraphx::shape::float_type, {3, 4}};
    auto al = p.add_literal(migraphx::literal{a_shape, a});
    migraphx::shape b_shape{migraphx::shape::float_type, {4, 3}};
    auto bl = p.add_literal(migraphx::literal{b_shape, b});
    migraphx::shape c_shape{migraphx::shape::float_type, {3, 3}};
    std::vector<float> vec_c(3 * 3, 4.0f);
    auto cl = p.add_literal(migraphx::literal{c_shape, c});
    p.add_instruction(migraphx::op::dot{alpha, beta}, al, bl, cl);

    return p;
}

migraphx::program create_program_vv(float alpha = 1.0f, float beta = 1.0f)
{
    migraphx::program p;
    std::vector<float> a = {1.49530002, -0.07181969,  0.44593846, -0.8645019 ,  0.52992304,
               -0.4910338 , -2.12179422, -0.45962977};
    std::vector<float> b = {1.13864253, -1.82376051,  1.06021781, -1.17140279, -0.68770616,
               -0.60046747, -0.7126266 ,  0.73346419};
    migraphx::shape a_shape{migraphx::shape::float_type, {8}};
    auto al = p.add_literal(migraphx::literal{a_shape, a});
    migraphx::shape b_shape{migraphx::shape::float_type, {8}};
    auto bl = p.add_literal(migraphx::literal{b_shape, b});
    p.add_instruction(migraphx::op::dot{alpha, beta}, al, bl);

    return p;
}

migraphx::program create_program_vm(float alpha = 1.0f, float beta = 1.0f)
{
    migraphx::program p;
    std::vector<float> a = {1.49530002, -0.07181969,  0.44593846, -0.8645019 ,  0.52992304,
               -0.4910338 , -2.12179422, -0.45962977};
    std::vector<float> b = {
	   -0.06210242,  0.0187149 ,  1.47482984, -1.19590602, -0.45601701,
        0.36934488, -0.83913193,  0.75350964,  0.80707019,  0.35923582,
       -2.18480722, -0.85608682,  0.75849199,  0.49103473, -0.91329477,
       -0.36364322, -0.69688937,  0.07165814, -0.15505523,  0.52221663,
       -0.98631192, -0.37353654, -1.89818706, -0.87209739, -0.33942003,
        0.11390353,  0.78181162, -0.18395337, -0.34743419, -0.08091231,
        1.21119765,  1.23869861,  1.42169414,  0.86412382,  1.05898002,
       -0.31918307,  1.08546695,  1.50682711, -0.66083538, -0.32683929};
    migraphx::shape a_shape{migraphx::shape::float_type, {8}};
    auto al = p.add_literal(migraphx::literal{a_shape, a});
    migraphx::shape b_shape{migraphx::shape::float_type, {8, 5}};
    auto bl = p.add_literal(migraphx::literal{b_shape, b});
    migraphx::shape c_shape{migraphx::shape::float_type};
    auto cl = p.add_literal(migraphx::literal{c_shape, {0}});
    p.add_instruction(migraphx::op::dot{alpha, beta}, al, bl);

    return p;
}

migraphx::program create_program_vbm(float alpha = 1.0f, float beta = 1.0f)
{
    migraphx::program p;
    std::vector<float> a = {-1.7468318 , -0.38900251,  1.00183915,  
		0.06016438,  0.08295905, 1.5830535};
    std::vector<float> b = {
		 1.2459538 ,  0.39586199, -0.77035574,  0.22689828,
         0.3289835 ,  1.02804361, -0.22941113, -0.33940324,
         0.80078249,  1.0319152 ,  0.80034948, -0.11631159,
         0.36899208, -0.28506697, -1.2211584 , -0.55678377,
        -0.3618498 ,  0.34857264, -0.38700147, -0.43434611,
         1.73029783, -0.71578372,  0.09777723,  0.06616614,
        -1.66721186, -0.16046032, -1.64581663,  1.09373609,
        -0.14127692, -0.01938473, -0.67310303, -1.56154787,
        -1.0665462 ,  0.68538535, -1.53920085, -0.35710272,
         0.06887234,  0.17474616,  1.08194804, -0.19990148,
        -0.91149488,  0.95303646,  0.95448717, -0.49332393,
        -1.762213  , -0.56571194, -1.69704968, -0.82798066,
         0.65531872,  1.5007798 ,  0.99877355,  0.53386114,
        -0.88150609, -1.0756985 ,  0.50962511, -0.68019002,
         0.1583068 ,  2.83988407, -1.10292457,  0.02126969,
         0.21129951,  0.25690146, -1.6490316 ,  0.55261771,
        -1.70504303, -0.02870394, -0.18205627,  0.29446203,
        -1.91360924,  0.46102174,  0.44977568, -0.48113321};

    migraphx::shape a_shape{migraphx::shape::float_type, {6}};
    auto al = p.add_literal(migraphx::literal{a_shape, a});
    migraphx::shape b_shape{migraphx::shape::float_type, {3, 6, 4}};
    auto bl = p.add_literal(migraphx::literal{b_shape, b});
    migraphx::shape c_shape{migraphx::shape::float_type};
    auto cl = p.add_literal(migraphx::literal{c_shape, {0}});
    p.add_instruction(migraphx::op::dot{alpha, beta}, al, bl);

    return p;
}

migraphx::program create_program_mv(float alpha = 1.0f, float beta = 1.0f)
{
    migraphx::program p;
    std::vector<float> a = {
        0.1612524 ,  0.61266466, -0.19212896,  1.34228825, -1.09746949,
        0.4680955 , -0.431748  , -0.89791241, -2.19078702, -0.13767058,
       -1.66105228, -0.91834613,  0.59199744,  1.41967261,  0.76237423};

    std::vector<float> b = {
0.14365572,  0.23401411, -0.8970094 , -0.12526676, -1.04703286};

    migraphx::shape a_shape{migraphx::shape::float_type, {3, 5}};
    auto al = p.add_literal(migraphx::literal{a_shape, a});
    migraphx::shape b_shape{migraphx::shape::float_type, {5}};
    auto bl = p.add_literal(migraphx::literal{b_shape, b});
    migraphx::shape c_shape{migraphx::shape::float_type};
    auto cl = p.add_literal(migraphx::literal{c_shape, {0}});
    p.add_instruction(migraphx::op::dot{alpha, beta}, al, bl);

    return p;
}

migraphx::program create_program_bmv(float alpha = 1.0f, float beta = 1.0f)
{
    migraphx::program p;
    std::vector<float> a = {
		  1.24593227, -0.84351316,  0.27882229, -0.42518484, -1.11391528,
          0.59141834,  1.34198714,  2.25884063, -1.32093452,  0.44766336,
         -0.09306479,  0.47526699,  0.25858488,  1.30820392,  1.17186787,
          0.31530864, -1.19159424, -0.24100903, -1.03857886,  1.54453427,
          0.05041654,  1.67108177,  0.965805  ,  0.52958924, -1.61243992,
          0.02941846,  0.77523836,  1.97963853, -2.51093596,  0.21882645,
         -2.60193574,  1.1899952 ,  1.70883519,  0.94586745,  2.65002512,
         -1.42427102,  1.0143951 , -1.34115312,  1.63833732, -1.46477355,
          0.44014877,  0.58032696, -1.63874372, -0.82834423,  1.81131778,
         -0.52393379,  1.16721943,  0.39488835,  0.23947128, -0.15733194,
          0.19451158,  1.21315445,  0.44594897,  0.40809135, -0.64252994,
          0.7541716 , -0.97203195,  0.69208485,  0.34350988,  0.9836842 };
    std::vector<float> b = {
        0.05013914,  1.39932885,  2.56616476,  1.02225623, -0.03977829};

    migraphx::shape a_shape{migraphx::shape::float_type, {2, 2, 3, 5}};
    auto al = p.add_literal(migraphx::literal{a_shape, a});
    migraphx::shape b_shape{migraphx::shape::float_type, {5}};
    auto bl = p.add_literal(migraphx::literal{b_shape, b});
    migraphx::shape c_shape{migraphx::shape::float_type};
    auto cl = p.add_literal(migraphx::literal{c_shape, {0}});
    p.add_instruction(migraphx::op::dot{alpha, beta}, al, bl);

    return p;
}

migraphx::program create_program_mm(float alpha = 1.0f, float beta = 1.0f)
{
    migraphx::program p;
    std::vector<float> a = {
        0.24627778, -0.03701134, -1.12684829,  0.1327002 , -0.46702924,
       -0.84460656,  0.20488264,  0.52745128,  0.291846  ,  0.04456201,
       -1.12557818,  0.96434542, -0.46159089,  0.283914  , -1.4594178 };
    std::vector<float> b = {
        0.49899375, -2.20168661,  1.08895066,
       -0.01135643,  0.90570669, -1.43550963,
       -1.73033377,  0.21338776,  0.96962508,
        0.38913968, -0.32822861,  0.88222863,
        0.93330718, -1.24265228, -1.62587164};

    migraphx::shape a_shape{migraphx::shape::float_type, {3, 5}};
    auto al = p.add_literal(migraphx::literal{a_shape, a});
    migraphx::shape b_shape{migraphx::shape::float_type, {5, 3}};
    auto bl = p.add_literal(migraphx::literal{b_shape, b});
    migraphx::shape c_shape{migraphx::shape::float_type};
    auto cl = p.add_literal(migraphx::literal{c_shape, {0}});
    p.add_instruction(migraphx::op::dot{alpha, beta}, al, bl);

    return p;
}

migraphx::program create_program_bmm(float alpha = 1.0f, float beta = 1.0f)
{
    migraphx::program p;
    std::vector<float> a = {
         -0.49450006, -1.07431991, -0.02796692, -0.99631927,  0.20040449,
         -1.39709437, -0.15695328,  0.08208373, -0.09746386,  0.77923021,
         -0.1849151 ,  0.14419043, -0.25798175, -0.2504807 , -1.11134383,
         -0.71030613, -0.20234025,  0.90229168,  0.62643053, -0.83512638,
          1.66051254,  0.05941673,  0.73081559,  0.27111867,  0.55060745,
          0.34999583,  1.02236619,  0.60178395,  1.49646162,  1.93255155,
         -3.65357913, -1.38059906, -0.46302398,  0.19847152,  0.39785875,
          1.47004861, -1.24482133, -0.01954702,  0.36073898,  1.56055978,
         -0.10344603, -0.34283135, -0.56482649,  1.80861249, -0.92268202,
          0.94371182, -0.02373232, -0.75441145,  0.43325034,  0.4057425 ,
         -0.48844822, -0.36390512,  0.74110406,  1.25158366,  0.52196654,
          1.43461691, -0.57530864, -0.66716206, -1.76516289,  0.96582849};
    std::vector<float> b = {
        0.49899375, -2.20168661,  1.08895066,
       -0.01135643,  0.90570669, -1.43550963,
       -1.73033377,  0.21338776,  0.96962508,
        0.38913968, -0.32822861,  0.88222863,
        0.93330718, -1.24265228, -1.62587164};

    migraphx::shape a_shape{migraphx::shape::float_type, {2, 2, 3, 5}};
    auto al = p.add_literal(migraphx::literal{a_shape, a});
    migraphx::shape b_shape{migraphx::shape::float_type, {5, 3}};
    auto bl = p.add_literal(migraphx::literal{b_shape, b});
    migraphx::shape c_shape{migraphx::shape::float_type};
    auto cl = p.add_literal(migraphx::literal{c_shape, {0}});
    p.add_instruction(migraphx::op::dot{alpha, beta}, al, bl);

    return p;
}

migraphx::program create_program_mbm(float alpha = 1.0f, float beta = 1.0f)
{
    migraphx::program p;
    std::vector<float> a = {
	   -0.0309568 , -1.57294749, -0.00768606,  1.5786921 ,
        0.50519718,  0.10530702, -0.05302112, -0.06503757,
        0.4079716 ,  0.0799132 , -0.82624962,  0.49341502};

    std::vector<float> b = {
          0.3664867 ,  0.24649534,  1.14728076,
          1.09911548, -1.23711247, -0.49436419,
         -0.67557879, -0.84180575, -1.09754376,
          0.07807351,  0.74349043, -0.92084701,
          0.50267885,  0.78709401,  0.80598159,
         -0.51269589, -0.40337193,  0.29457878,
          1.25447301, -1.66251457, -1.54652239,
         -0.35067765, -0.5214464 , -0.7866878 ,
          1.11128573,  0.26927291, -0.0929818 ,
          0.07523954,  0.3256776 , -1.08617826,
          0.89294253, -0.91007619, -2.42825765,
         -1.76805581,  1.08136334, -0.14521253,
         -1.32061148,  0.60663124, -1.19835255,
         -0.98803563, -1.06927896, -0.51967419,
         -0.98974639,  1.01287011,  1.34910394,
          0.1203349 ,  0.67387452, -0.32447465,
          1.15187449, -0.82253807,  0.22302433,
          0.46434695,  0.319647  ,  1.56459445,
          0.15664012,  0.03998102,  0.62981041,
          0.11831296,  0.47824434, -0.93941882,
         -0.34674036,  1.17071104,  0.59203806,
          2.75817738, -0.69300013,  1.30971899,
         -0.14231862, -1.90915568, -0.06895489,
          0.20160375,  0.01945916,  0.03586956};

    migraphx::shape a_shape{migraphx::shape::float_type, {3, 4}};
    auto al = p.add_literal(migraphx::literal{a_shape, a});
    migraphx::shape b_shape{migraphx::shape::float_type, {2, 3, 4, 3}};
    auto bl = p.add_literal(migraphx::literal{b_shape, b});
    p.add_instruction(migraphx::op::dot{alpha, beta}, al, bl);

    return p;
}

migraphx::program create_program_bmbm1(float alpha = 1.0f, float beta = 1.0f)
{
    migraphx::program p;
    std::vector<float> a = {
         -0.49450006, -1.07431991, -0.02796692, -0.99631927,  0.20040449,
         -1.39709437, -0.15695328,  0.08208373, -0.09746386,  0.77923021,
         -0.1849151 ,  0.14419043, -0.25798175, -0.2504807 , -1.11134383,
         -0.71030613, -0.20234025,  0.90229168,  0.62643053, -0.83512638,
          1.66051254,  0.05941673,  0.73081559,  0.27111867,  0.55060745,
          0.34999583,  1.02236619,  0.60178395,  1.49646162,  1.93255155,
         -3.65357913, -1.38059906, -0.46302398,  0.19847152,  0.39785875,
          1.47004861, -1.24482133, -0.01954702,  0.36073898,  1.56055978,
         -0.10344603, -0.34283135, -0.56482649,  1.80861249, -0.92268202,
          0.94371182, -0.02373232, -0.75441145,  0.43325034,  0.4057425 ,
         -0.48844822, -0.36390512,  0.74110406,  1.25158366,  0.52196654,
          1.43461691, -0.57530864, -0.66716206, -1.76516289,  0.96582849};
    std::vector<float> b = {
		 -1.12211357,  1.74720423,  0.60382572,
         -0.61090125, -0.3315936 ,  0.30924675,
         -0.28906435,  0.64039247, -1.2822253 ,
          0.55899286,  2.14013013,  1.00944809,
          0.21660017, -0.75465098,  0.12097934,
         -1.64006315,  0.43582108, -0.64348541,
          0.43101069,  1.30191386,  1.7746011 ,
          0.24935804,  0.42830791, -0.13593643,
          0.38749427,  1.39776254, -0.42911717,
         -1.3537624 , -0.81999648, -0.1754485 };
    migraphx::shape a_shape{migraphx::shape::float_type, {2, 2, 3, 5}};
    auto al = p.add_literal(migraphx::literal{a_shape, a});
    migraphx::shape b_shape{migraphx::shape::float_type, {2, 1, 5, 3}};
    auto bl = p.add_literal(migraphx::literal{b_shape, b});
    migraphx::shape c_shape{migraphx::shape::float_type, {2, 2, 3, 3}};
	std::vector<float> c = {
          0.70574512, -2.80915314, -1.57644969,
          1.75415381, -3.13303087, -1.00150259,
         -0.18675123, -0.23349122, -0.12357225,
          0.82911538,  1.37473744, -1.11709934,
         -1.84001907,  3.51427391,  0.42425673,
          0.0638482 ,  2.40210271,  1.50027643,
          4.81988916, -3.63687142, -0.19101717,
         -4.92522092, -1.76377022, -3.58095615,
          1.83096922,  2.5512663 , -1.07926588,
         -2.12749134,  0.33014536, -0.80393025,
          0.60740202,  0.95217761, -1.06087445,
         -4.75868152, -3.6687713 , -1.26539821};
    auto cl = p.add_literal(migraphx::literal{c_shape, {0}});
    p.add_instruction(migraphx::op::dot{alpha, beta}, al, bl);

    return p;
}

migraphx::program create_program_bmbm2(float alpha = 1.0f, float beta = 1.0f)
{
    migraphx::program p;
    std::vector<float> a = {
		 -0.19276159, -1.2568421 , -0.321242  ,  1.21471077, -0.4927751 ,
          0.69446894, -0.1786371 , -1.00763473, -0.10279314,  3.02931355,
          1.08359235, -0.35190132, -0.00639111,  0.78989113,  1.23538029,
          0.4590747 ,  0.17304142,  0.42512412,  0.21076913, -0.01724556,
         -0.17763898,  0.12852236, -0.00459301,  1.34498824,  0.02907823,
          0.1784464 , -0.20790355, -0.52336699,  0.45804085,  1.06025801};

    std::vector<float> b = {
		 -1.12211357,  1.74720423,  0.60382572,
         -0.61090125, -0.3315936 ,  0.30924675,
         -0.28906435,  0.64039247, -1.2822253 ,
          0.55899286,  2.14013013,  1.00944809,
          0.21660017, -0.75465098,  0.12097934,
         -1.64006315,  0.43582108, -0.64348541,
          0.43101069,  1.30191386,  1.7746011 ,
          0.24935804,  0.42830791, -0.13593643,
          0.38749427,  1.39776254, -0.42911717,
         -1.3537624 , -0.81999648, -0.1754485 };
    migraphx::shape a_shape{migraphx::shape::float_type, {1, 2, 3, 5}};
    auto al = p.add_literal(migraphx::literal{a_shape, a});
    migraphx::shape b_shape{migraphx::shape::float_type, {2, 1, 5, 3}};
    auto bl = p.add_literal(migraphx::literal{b_shape, b});
    migraphx::shape c_shape{migraphx::shape::float_type, {2, 2, 3, 3}};
	std::vector<float> c = {
           1.64924590e+00,   2.84575831e+00,   1.07340773e+00,
           2.19817080e-01,  -1.87873283e+00,   1.91883003e+00,
          -2.89962196e-01,   2.76404142e+00,   1.50048102e+00,
          -6.29650347e-01,   1.48105185e+00,  -3.71716505e-03,
           8.80281500e-01,   2.50057585e+00,   1.29958508e+00,
           5.63751779e-01,   2.25703781e-01,   1.30516919e+00,
           8.32118386e-01,   2.44050864e-01,  -2.49748221e+00,
          -5.60803176e+00,  -2.98919069e+00,  -1.11429417e+00,
          -3.29675989e+00,   1.02442564e-01,  -1.87659303e+00,
          -4.67302454e-01,   9.16189968e-01,  -1.33537175e-01,
           8.27398578e-01,   1.94406914e+00,  -2.39250915e-01,
          -1.77062701e+00,  -6.46239534e-01,  -7.95202750e-01};
    auto cl = p.add_literal(migraphx::literal{c_shape, {0}});
    p.add_instruction(migraphx::op::dot{alpha, beta}, al, bl);

    return p;
}

migraphx::program create_program_bmbm3(float alpha = 1.0f, float beta = 1.0f)
{
    migraphx::program p;
    std::vector<float> a = {
         -0.55248691,  0.70275958,  0.56967633,  0.88206033,
         -0.85088547,  0.05689149, -0.20084703,  0.18024434,
          1.0730491 ,  0.15913531,  0.93621628,  0.35072771,
          1.28616952,  1.55384379,  0.30376261, -1.12356544,
         -0.64271552, -2.50703079, -0.23994372,  0.8166084 ,
          0.06542249, -0.17472336, -0.37665211,  0.16342699,
          0.07645941,  0.65024333, -1.19883423, -0.40536776,
         -0.31132765,  0.78113691, -0.16887638,  2.30797418,
         -0.36241233,  0.33552153, -1.05343996, -0.16909699,
         -1.22608815,  1.64165613,  0.96260828, -0.16733976,
          0.84211199,  1.31243813,  0.89258549, -0.48250384,
         -1.06005206,  1.37021342, -0.35658565,  0.26879188};

    std::vector<float> b = {
          0.17111129, -0.82134741, -1.58001178, -1.46759447,  0.31522514,
         -0.11567352, -0.038978  , -0.3601414 , -0.84379876,  0.24848939,
         -0.37080544,  0.00838631,  1.51316241,  0.42385344,  2.06043846,
          1.82348849,  1.07180434,  0.6567393 ,  1.41164561,  0.73091185,
         -0.33541302, -0.98082287, -0.06605479,  0.82219717, -1.41619634,
          0.51326658,  0.26916313,  0.79819769,  0.85583702,  0.07876046,
         -0.42375545, -0.7758751 ,  1.14334296, -0.14211708, -1.54520411,
         -0.55244869, -0.48478899,  0.10782164, -0.20879552, -0.99019754,
          1.78783102, -1.31610052,  1.73510175, -0.48360172,  0.62367417,
         -1.34180545, -0.37512931, -1.50521357,  0.08383314,  0.76165608,
         -0.4961646 ,  0.95821311, -0.68407191,  0.48299435, -0.24323988,
          0.34793412,  0.37908669,  1.19083454,  1.30218795, -0.26731035,
         -0.34544132, -0.09595373,  0.50951334,  0.48896956,  0.38753818,
         -0.4939919 ,  0.02352126,  0.42013764,  0.07027765,  0.21169851,
         -0.24411376, -1.77793736, -0.88370924,  0.95294025, -0.08208804,
         -0.95943892,  0.30280474,  1.1967013 , -1.17700948,  0.29533973};
    migraphx::shape a_shape{migraphx::shape::float_type, {2, 2, 3, 4}};
    auto al = p.add_literal(migraphx::literal{a_shape, a});
    migraphx::shape b_shape{migraphx::shape::float_type, {2, 2, 4, 5}};
    auto bl = p.add_literal(migraphx::literal{b_shape, b});
    migraphx::shape c_shape{migraphx::shape::float_type, {2, 2, 3, 5}};
	std::vector<float> c = {
          1.22136035,  1.3765651 ,  2.0611395 ,  1.70445494,  1.8189619 ,
          0.2509717 ,  0.88815736,  1.13837946,  1.37006127, -0.53617378,
          0.45759693, -0.503786  , -0.10575749, -0.81715738,  2.56316255,
          0.85812927, -0.53425671,  1.38147704,  2.57874755, -1.05591061,
         -1.42065674, -0.25412658, -2.14494165, -2.81045272,  0.27491485,
         -0.04229986,  0.10181043, -0.55680682, -0.07633866,  0.313767  ,
         -0.28202571, -1.64696179, -0.50872733, -1.08935912,  0.94291084,
         -0.71792156,  0.82981387,  1.14797592,  3.13989358, -0.17507726,
         -0.63429162, -0.72241531, -0.61459168, -0.52561056,  0.3309648 ,
         -0.46185697, -1.60586695, -0.98590829,  0.63012062, -0.25606052,
         -0.69419352, -1.78299913, -0.38572706,  1.92249442,  0.3884186 ,
         -0.48153048,  0.84932351,  0.67234919, -1.07821322, -0.01208216};
    auto cl = p.add_literal(migraphx::literal{c_shape, {0}});
    p.add_instruction(migraphx::op::dot{alpha, beta}, al, bl);

    return p;
}

migraphx::program create_program_bmbm4(float alpha = 1.0f, float beta = 1.0f)
{
    migraphx::program p;
    std::vector<float> a = {
         -0.55248691,  0.70275958,  0.56967633,  0.88206033,
         -0.85088547,  0.05689149, -0.20084703,  0.18024434,
          1.0730491 ,  0.15913531,  0.93621628,  0.35072771,
          1.28616952,  1.55384379,  0.30376261, -1.12356544,
         -0.64271552, -2.50703079, -0.23994372,  0.8166084 ,
          0.06542249, -0.17472336, -0.37665211,  0.16342699,
          0.07645941,  0.65024333, -1.19883423, -0.40536776,
         -0.31132765,  0.78113691, -0.16887638,  2.30797418,
         -0.36241233,  0.33552153, -1.05343996, -0.16909699,
         -1.22608815,  1.64165613,  0.96260828, -0.16733976,
          0.84211199,  1.31243813,  0.89258549, -0.48250384,
         -1.06005206,  1.37021342, -0.35658565,  0.26879188};

    std::vector<float> b = {
        -0.33734601,  0.66386073,  0.41425048,  0.40190389, -0.99645073,
        -0.10017067, -0.58542118,  0.48636962,  0.06301405,  1.14669128,
        -0.06526677,  0.23172741, -1.49693143, -0.44464233, -0.12775566,
        -1.32038007,  1.1812471 ,  1.22362746, -0.49013843,  0.25339836,
         1.31698705,  1.54256669,  0.11211132, -0.18005487,  0.36730145,
         0.97705953, -0.18909084,  0.544932  ,  0.32891878,  0.64250015,
        -0.41381398,  0.47402562,  1.22286761,  1.07573211, -0.92988077,
        -0.36340925, -1.76152377, -0.96642674, -0.79231929,  0.11517073};

    migraphx::shape a_shape{migraphx::shape::float_type, {2, 2, 3, 4}};
    auto al = p.add_literal(migraphx::literal{a_shape, a});
    migraphx::shape b_shape{migraphx::shape::float_type, {2, 4, 5}};
    auto bl = p.add_literal(migraphx::literal{b_shape, b});
    migraphx::shape c_shape{migraphx::shape::float_type, {2, 2, 3, 5}};
	std::vector<float> c = {
         -1.08585245,  0.39575611,  0.33947977, -0.86339678,  1.50710753,
          0.05646156, -0.43180359,  0.19639674, -0.33742881,  0.98443538,
         -0.9021272 ,  1.25043704, -0.45038184, -0.14689614, -0.91749459,
          3.49467934,  3.81336312,  2.4482385 ,  1.49649707,  1.05889193,
         -3.49343731, -2.06958956, -2.52082858, -1.61401519, -1.52966956,
          0.01191848, -0.33246613, -0.70641362, -0.60391255,  0.28083355,
          0.52255496, -1.08655006,  1.64648546,  0.80344255,  0.71987865,
         -3.00960296,  2.02318221,  3.32785057, -1.13203844,  1.81235734,
          0.38067585, -0.88086897,  1.38307367,  0.42677257,  0.83759966,
         -0.34827442, -1.45067092,  2.09599671,  1.92882983, -0.30996324,
          2.19736278,  2.32389426,  2.36741832,  1.62253915,  0.26698225,
         -0.00741609, -2.53680983, -0.0679954 ,  0.04499683,  0.85354276};
    auto cl = p.add_literal(migraphx::literal{c_shape, {0}});
    p.add_instruction(migraphx::op::dot{alpha, beta}, al, bl);

    return p;
}

migraphx::program create_program_bmbm5(float alpha = 1.0f, float beta = 1.0f)
{
    migraphx::program p;
    std::vector<float> a = {
         0.44434486, -0.4775394 ,  1.22403495,  1.3390557 ,
        -0.16682514, -0.14706984,  2.03517409, -0.15236999,
         1.31615472, -0.98724552,  0.87351608,  0.32548614,
        -3.41102373, -1.98384933,  0.50167115,  0.59746381,
         0.52601494,  1.68033723, -1.69118135, -0.07171001,
        -0.21904557, -0.1435285 , -0.3086262 , -0.55035202};

    std::vector<float> b = {
         -0.94363619, -0.77647765, -0.67011854, -2.09503007,
          0.90123996, -0.46622586,  1.42071249,  0.03609514,
          0.15959348,  1.39677643,  1.04978928,  1.00156894,
         -0.27378851,  0.0874493 ,  1.34600448,  2.08173849,
          0.46533488,  0.00631963, -0.56208786,  0.02443816,
          0.45989363,  0.62163606, -0.4031336 ,  0.46017999,
          0.39662946, -0.47854661,  1.67630842, -0.21867977,
          0.63853741,  0.45437104,  0.29735596, -0.71734146,
          0.1237553 ,  0.0409191 ,  0.14675446, -0.28671886,
         -0.10558661,  0.45182015,  0.52462527,  0.85523901,
         -0.99229207,  0.35318084, -1.00044197,  1.79608682,
         -0.45742108, -0.70323029, -0.39590981, -0.46266041,
         -0.69778675,  0.37064368,  0.47614881, -0.30574358,
          0.51562266,  1.47646532,  0.81795032,  0.62790241,
         -1.17363991, -0.82171213,  0.43211813, -0.63605139,
          1.18437641,  0.23012845, -0.37945211,  0.01256212};

    migraphx::shape a_shape{migraphx::shape::float_type, {2, 3, 4}};
    auto al = p.add_literal(migraphx::literal{a_shape, a});
    migraphx::shape b_shape{migraphx::shape::float_type, {2, 2, 4, 4}};
    auto bl = p.add_literal(migraphx::literal{b_shape, b});
    migraphx::shape c_shape{migraphx::shape::float_type, {2, 2, 3, 4}};
	std::vector<float> c = {
         -1.02094755,  1.70442001,  2.1111438 ,  3.06536646,
          0.39139469,  3.0274623 ,  1.83426191,  2.06536787,
         -2.08142323,  0.68688487, -0.92945811, -1.2405549 ,
         -1.91914741, -1.22339147,  3.73566635, -1.5345778 ,
          0.30098761,  1.82460858, -3.82933195,  1.20738012,
         -0.64176798, -0.19297878, -0.50001913,  0.39087862,
         -1.72170067, -0.70693856, -1.94004086,  1.0431326 ,
         -1.95490676,  0.75266023, -2.07738769,  3.64789696,
         -0.74854627, -0.31258412, -1.32754766,  0.1966239 ,
          1.47609026, -4.46809498, -3.2567728 , -0.51434837,
          2.39927998,  4.04908547,  0.92131416,  1.96903951,
         -0.21076738, -0.16615248, -0.1462282 ,  0.16623842};

    auto cl = p.add_literal(migraphx::literal{c_shape, {0}});
    p.add_instruction(migraphx::op::dot{alpha, beta}, al, bl);

    return p;
}

migraphx::program create_program_vv2(float alpha = 1.0f, float beta = 1.0f)
{
    migraphx::program p;
    migraphx::shape m1_shape{migraphx::shape::float_type, {5}};
    migraphx::shape m2_shape{migraphx::shape::float_type, {5}};
    auto l1 = p.add_parameter("1", m1_shape);
    auto l2 = p.add_parameter("2", m2_shape);

    p.add_instruction(migraphx::op::dot{}, l1, l2);

    return p;
}

migraphx::program create_program_gemm_3args(float alpha = 1.0f, float beta = 1.0f)
{
    migraphx::program p;
    migraphx::shape m1_shape{migraphx::shape::float_type, {1, 4096}};
    migraphx::shape m2_shape{migraphx::shape::float_type, {1000, 4096}};
    migraphx::shape m3_shape{migraphx::shape::float_type, {1, 1000}};
    std::vector<float> data1(4096, 1.0f);
    std::vector<float> data2(4096 * 1000, 1.0f);
    std::vector<float> data3(1000, 1.0f);


    auto l1 = p.add_literal(migraphx::literal{m1_shape, data1});
    auto l2 = p.add_literal(migraphx::literal{m2_shape, data2});
    auto tl2 = p.add_instruction(migraphx::op::transpose{{1, 0}}, l2);
    auto l3 = p.add_literal(migraphx::literal{m3_shape, data3});

    p.add_instruction(migraphx::op::dot{alpha, beta}, l1, tl2);

    return p;
}

migraphx::program create_program_gemm_3args1(float alpha = 1.0f, float beta = 1.0f)
{
    migraphx::program p;
    migraphx::shape m1_shape{migraphx::shape::float_type, {8, 1}};
    migraphx::shape m2_shape{migraphx::shape::float_type, {4, 8}};
    migraphx::shape m3_shape{migraphx::shape::float_type, {1, 4}};
    std::vector<float> data1(8, 1.0f);
    std::vector<float> data2(8 * 4, 1.0f);
    std::vector<float> data3(4, 1.0f);


    auto l1 = p.add_literal(migraphx::literal{m1_shape, data1});
    auto l2 = p.add_literal(migraphx::literal{m2_shape, data2});
    //auto tl2 = p.add_instruction(migraphx::op::transpose{{1, 0}}, l2);
    auto l3 = p.add_literal(migraphx::literal{m3_shape, data3});

    p.add_instruction(migraphx::op::dot{alpha, beta}, l2, l1);

    return p;
}

migraphx::program create_relu_program()
{
    migraphx::program p;
    std::size_t axis = 0;
    migraphx::shape s0{migraphx::shape::float_type, {2, 2}};
    migraphx::shape s1{migraphx::shape::float_type, {3, 2}};
    migraphx::shape s2{migraphx::shape::float_type, {1, 2}};
    auto l0 = p.add_parameter("x", s0);
    auto l1 = p.add_parameter("y", s1);
    auto l2 = p.add_parameter("z", s2);
    auto r0 = p.add_instruction(migraphx::op::relu{}, l0);
    auto r1 = p.add_instruction(migraphx::op::relu{}, l1);
    auto r2 = p.add_instruction(migraphx::op::relu{}, l2);
    auto c0 = p.add_instruction(migraphx::op::concat{axis}, r0, r1, r2);
    p.add_instruction(migraphx::op::relu{}, c0);
    return p;
}

migraphx::program create_lstm_program() 
{
    std::size_t batch_size  = 2;
    std::size_t seq_len     = 3;
    std::size_t hidden_size = 5;
    std::size_t input_size  = 8;
    std::size_t num_dirct   = 1;
    float clip              = 0.0f;

    migraphx::program p;
    migraphx::shape in_shape{migraphx::shape::float_type, {seq_len, batch_size, input_size}};
    migraphx::shape w_shape{migraphx::shape::float_type,
                            {num_dirct, 4 * hidden_size, input_size}};
    migraphx::shape r_shape{migraphx::shape::float_type,
                            {num_dirct, 4 * hidden_size, hidden_size}};
    migraphx::shape b_shape{migraphx::shape::float_type, {num_dirct, 8 * hidden_size}};
    migraphx::shape ih_shape{migraphx::shape::float_type, {num_dirct, batch_size, hidden_size}};
    migraphx::shape ic_shape{migraphx::shape::float_type, {num_dirct, batch_size, hidden_size}};
    migraphx::shape pph_shape{migraphx::shape::float_type, {num_dirct, 3 * hidden_size}};

    auto seq  = p.add_parameter("seq", in_shape);
    auto w    = p.add_parameter("w", w_shape);
    auto r    = p.add_parameter("r", r_shape);
    auto bias = p.add_parameter("bias", b_shape);
    auto ih   = p.add_parameter("ih", ih_shape);
    auto ic   = p.add_parameter("ic", ic_shape);
    auto pph  = p.add_parameter("pph", pph_shape);
    auto und  = p.add_instruction(migraphx::op::undefined{});

    p.add_instruction(
        migraphx::op::lstm{
            hidden_size,
            {migraphx::op::sigmoid{}, migraphx::op::tanh{}, migraphx::op::tanh{}},
            migraphx::op::rnn_direction::forward,
            clip},
        seq,
        w,
        r,
        bias,
        und,
        ih,
        ic,
        pph);

    return p;
}

migraphx::program create_relu_lrn_program()
{
    migraphx::program p;
    auto x = p.add_parameter("x", migraphx::shape{migraphx::shape::float_type, {1, 5, 2, 2}});
    auto ty = p.add_instruction(migraphx::op::transpose{{0, 2, 3, 1}}, x);
    auto l = p.add_instruction(migraphx::op::tanh{}, ty);
    p.add_instruction(migraphx::op::add{}, l, l);
    return p;
}

migraphx::program create_tanh_program() 
{
    migraphx::program p;
    migraphx::shape s{migraphx::shape::float_type, {4, 3}};
    std::vector<float> data(4 * 3);
    std::iota(data.begin(), data.end(), 0.0f);
    //auto x = p.add_parameter("x", migraphx::shape{migraphx::shape::float_type, {4, 3}});
    auto x = p.add_literal(migraphx::literal(s, data));
    auto tx = p.add_instruction(migraphx::op::transpose{{1, 0}}, x);
    auto tanhx = p.add_instruction(migraphx::op::sin{}, tx);
    //p.add_instruction(migraphx::op::add{}, tanhx, tanhx);
    return p;
}

migraphx::program create_program_ladd()
{
    migraphx::program p;
    migraphx::shape s{migraphx::shape::float_type, {2, 3}};
    std::vector<float> data(2 * 3);
    std::iota(data.begin(), data.end(), 1.0f);
    auto l1 = p.add_literal(migraphx::literal(s, data));
    auto l2 = p.add_literal(migraphx::literal(s, data));
    p.add_instruction(migraphx::op::add{}, l1, l2);
    //migraphx::quantize(p, {"all"});
    return p;
};

migraphx::program create_lrn_program()
{
    migraphx::program p;
    auto x = p.add_parameter("x", migraphx::shape{migraphx::shape::float_type, {1, 5, 2, 2}});
    auto y = p.add_instruction(migraphx::op::relu{}, x);
    p.add_instruction(migraphx::op::lrn{0.0001, 0.75, 2.0, 5}, y);
    return p;
}


int main()
{
    auto p = create_lrn_program();
    std::vector<int> cpu_res, gpu_res;
    run_cpu(p, cpu_res);
    run_gpu(p, gpu_res);

    bool b_res = compare_results(cpu_res, gpu_res);

    //std::cout << "cpu_res = " << std::endl;
    //for_each(cpu_res.begin(), cpu_res.end(), [](auto f) {std::cout << f << "\t";});
    //std::cout << std::endl;

    //std::cout << "gpu_res = " << std::endl;
    //for_each(gpu_res.begin(), gpu_res.end(), [](auto f) {std::cout << f << "\t";});
    //std::cout << std::endl;

    std::cout << (b_res ? "PASSED!" : "FAILED") << std::endl;
    return 0;
}

