cmake_minimum_required(VERSION 3.5)
project (Test_migraphx)



set (MIGRAPHFOLDER /home/scxiao/Workplace/projects/AMDMIGraphX)
set (MIGRAPHLIBPATH /home/scxiao/Workplace/software/migraphlibs)
set (CMAKE_CXX_COMPILER /opt/rocm/bin/hcc)
set (CMAKE_CXX_STANDARD 14)
set (DEBUG _debug)

add_custom_target(
    copylibs
	COMMAND cp ${MIGRAPHFOLDER}/build${DEBUG}/src/onnx/libmigraphx_onnx.so          ${MIGRAPHLIBPATH}/.
	COMMAND cp ${MIGRAPHFOLDER}/build${DEBUG}/src/libmigraphx.so                    ${MIGRAPHLIBPATH}/.
	COMMAND cp ${MIGRAPHFOLDER}/build${DEBUG}/src/targets/cpu/libmigraphx_cpu.so    ${MIGRAPHLIBPATH}/.
	COMMAND cp ${MIGRAPHFOLDER}/build${DEBUG}/src/targets/gpu/libmigraphx_gpu.so    ${MIGRAPHLIBPATH}/.
	COMMAND cp ${MIGRAPHFOLDER}/build${DEBUG}/src/targets/gpu/libmigraphx_device.so ${MIGRAPHLIBPATH}/.)


include_directories(${MIGRAPHFOLDER}/src/include
                    ${MIGRAPHFOLDER}/src/targets/gpu/include
                    ${MIGRAPHFOLDER}/src/targets/cpu/include
                    ${MIGRAPHFOLDER}/test/include
                    ${MIGRAPHFOLDER}/deps_py/include)

link_directories(${MIGRAPHFOLDER}/build${DEBUG}/src/onnx 
                 ${MIGRAPHFOLDER}/build${DEBUG}/src
                 ${MIGRAPHFOLDER}/build${DEBUG}/src/targets/gpu
                 ${MIGRAPHFOLDER}/build${DEBUG}/src/targets/cpu
                 ${MIGRAPHFOLDER}/deps_py/lib)

file(GLOB util_SRC "*.cpp")
file(GLOB test_examples test_*.cpp)
file(GLOB util_HDR "*.hpp")
list(REMOVE_ITEM util_SRC ${test_examples})

add_library(test_util SHARED ${util_SRC})

foreach(filepath_name ${test_examples})
    get_filename_component(src_name ${filepath_name} NAME)
    get_filename_component(bin_name ${src_name} NAME_WE)
    message("source file: " ${src_name} " ---> bin: " ${bin_name})
    add_executable(${bin_name} ${src_name})
    target_link_libraries(${bin_name} migraphx migraphx_cpu migraphx_device migraphx_gpu migraphx_onnx hip_hcc MIOpen test_util)
#    add_dependencies(${bin_name} copylibs)
endforeach(src_name)

