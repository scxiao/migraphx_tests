cmake_minimum_required(VERSION 3.5)
project (Test_migraphx)



set (MIGRAPHX_FOLDER /home/scxiao/Workplace/projects/AMDMIGraphX)
#set (MIGRAPHX_FOLDER /home/scxiao/Workplace/projects/MIGraph)
set (DEPS_FOLDER deps_rocm3.3)
#set (DEPS_FOLDER deps_miopen2.2)
#set (MIGRAPHX_BUILD ${MIGRAPHX_FOLDER}/build_rocm3.0)
set (MIGRAPHX_BUILD ${MIGRAPHX_FOLDER}/build_debug)
#set (MIGRAPHX_BUILD ${MIGRAPHX_FOLDER}/build)
set (MIGRAPHX_DEPS /home/scxiao/Workplace/projects/AMDMIGraphX/${DEPS_FOLDER}/lib
                   /opt/rocm/lib)
set (CMAKE_CXX_COMPILER /opt/rocm/bin/hcc)
set (CMAKE_CXX_STANDARD 14)

add_custom_target(
    copylibs
	COMMAND cp ${MIGRAPHX_BUILD}/src/onnx/libmigraphx_onnx.so          ${MIGRAPHX_LIB_PATH}/.
	COMMAND cp ${MIGRAPHX_BUILD}/src/libmigraphx.so                    ${MIGRAPHX_LIB_PATH}/.
	COMMAND cp ${MIGRAPHX_BUILD}/src/targets/cpu/libmigraphx_cpu.so    ${MIGRAPHX_LIB_PATH}/.
	COMMAND cp ${MIGRAPHX_BUILD}/src/targets/gpu/libmigraphx_gpu.so    ${MIGRAPHX_LIB_PATH}/.
	COMMAND cp ${MIGRAPHX_BUILD}/src/targets/gpu/libmigraphx_device.so ${MIGRAPHX_LIB_PATH}/.)


include_directories(${MIGRAPHX_FOLDER}/src/include
                    ${MIGRAPHX_FOLDER}/src/api/include
                    ${MIGRAPHX_FOLDER}/src/targets/gpu/include
                    ${MIGRAPHX_FOLDER}/src/targets/cpu/include
                    ${MIGRAPHX_FOLDER}/test/include
                    /home/scxiao/Workplace/projects/AMDMIGraphX/${DEPS_FOLDER}/include)

#link_directories(${MIGRAPHX_BUILD}/src/onnx 
#                 ${MIGRAPHX_BUILD}/src
#                 ${MIGRAPHX_BUILD}/src/targets/gpu
#                 ${MIGRAPHX_BUILD}/src/targets/cpu
#                 ${MIGRAPHX_DEPS})

link_directories(${MIGRAPHX_BUILD}/lib
                 ${MIGRAPHX_DEPS})

file(GLOB util_SRC "*.cpp")
file(GLOB test_examples test_*.cpp)
file(GLOB util_HDR "*.hpp")
list(REMOVE_ITEM util_SRC ${test_examples})
set(test_examples test_rewrite_batchnorm.cpp)
#set(test_examples test_rnn.cpp)
#set(test_examples test_load_onnx.cpp test_run_onnx.cpp test_int8.cpp)

add_library(test_util SHARED ${util_SRC})

foreach(filepath_name ${test_examples})
    get_filename_component(src_name ${filepath_name} NAME)
    get_filename_component(bin_name ${src_name} NAME_WE)
    message("source file: " ${src_name} " ---> bin: " ${bin_name})
    add_executable(${bin_name} ${src_name})
    target_link_libraries(${bin_name} migraphx migraphx_cpu migraphx_device migraphx_gpu migraphx_onnx migraphx_c hip_hcc MIOpen test_util pthread)
#    add_dependencies(${bin_name} copylibs)
endforeach(filepath_name)

