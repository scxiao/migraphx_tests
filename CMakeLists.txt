cmake_minimum_required(VERSION 3.5)
project (Test_migraphx)


set (MIGRAPHFOLDER /home/scxiao/Workplace/projects/AMDMIGraphX)
set (CMAKE_CXX_COMPILER /opt/rocm/bin/hcc)
set(CMAKE_CXX_STANDARD 14)
include_directories(${MIGRAPHFOLDER}/src/include
                    ${MIGRAPHFOLDER}/src/targets/gpu/include
                    ${MIGRAPHFOLDER}/src/targets/cpu/include
                    ${MIGRAPHFOLDER}/test/include
                    ${MIGRAPHFOLDER}/deps_py/include)

set (DEBUG _debug)
link_directories(${MIGRAPHFOLDER}/build${DEBUG}/src/onnx 
                 ${MIGRAPHFOLDER}/build${DEBUG}/src
                 ${MIGRAPHFOLDER}/build${DEBUG}/src/targets/gpu
                 ${MIGRAPHFOLDER}/build${DEBUG}/src/targets/cpu
                 ${MIGRAPHFOLDER}/deps_py/lib)

file(GLOB test_examples *.cpp)
message(${test_examples})
foreach(filepath_name ${test_examples})
    get_filename_component(src_name ${filepath_name} NAME)
    message(${src_name})
    get_filename_component(bin_name ${src_name} NAME_WE)
    message(${bin_name})
    add_executable(${bin_name} ${src_name})
    target_link_libraries(${bin_name} migraphx migraphx_cpu migraphx_device migraphx_gpu migraphx_onnx hip_hcc MIOpen)
endforeach(src_name)

