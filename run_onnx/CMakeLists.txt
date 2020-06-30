cmake_minimum_required(VERSION 3.5)
project (Test_migraphx)



set (MIGRAPHX_FOLDER /home/scxiao/Workplace/projects/AMDMIGraphX)
set (MIGRAPHX_BUILD ${MIGRAPHX_FOLDER}/build)
set (MIGRAPHX_DEPS /home/scxiao/Workplace/projects/AMDMIGraphX/deps_mo1.8/lib)
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
                    ${MIGRAPHX_FOLDER}/src/targets/gpu/include
                    ${MIGRAPHX_FOLDER}/src/targets/cpu/include
                    ${MIGRAPHX_FOLDER}/test/include
                    /home/scxiao/Workplace/projects/AMDMIGraphX/deps_mo1.8/include)

link_directories(${MIGRAPHX_BUILD}/src/onnx 
                 ${MIGRAPHX_BUILD}/src
                 ${MIGRAPHX_BUILD}/src/targets/gpu
                 ${MIGRAPHX_BUILD}/src/targets/cpu
                 ${MIGRAPHX_DEPS})

file(GLOB util_HDR "*.hpp")
set(test_examples run_onnx.cpp)

foreach(filepath_name ${test_examples})
    get_filename_component(src_name ${filepath_name} NAME)
    get_filename_component(bin_name ${src_name} NAME_WE)
    message("source file: " ${src_name} " ---> bin: " ${bin_name})
    add_executable(${bin_name} ${src_name})
    target_link_libraries(${bin_name} migraphx migraphx_cpu migraphx_device migraphx_gpu migraphx_onnx hip_hcc MIOpen)
endforeach(filepath_name)

