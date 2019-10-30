cmake_minimum_required(VERSION 3.5)
project (Seq2seq_Translation)

set (MIGRAPHX_FOLDER /home/scxiao/Workplace/projects/AMDMIGraphX)
set (DEPS_FOLDER ${MIGRAPHX_FOLDER}/deps_rocblas2.7)
#set (MIGRAPHX_BUILD  ${MIGRAPHX_FOLDER}/build_debug)
set (MIGRAPHX_BUILD  ${MIGRAPHX_FOLDER}/build)
set (CMAKE_CXX_COMPILER /opt/rocm/bin/hcc)
set (CMAKE_CXX_STANDARD 14)

add_definitions(-DBLIBCXX_DEBUG)
message("current folder is: " ${PROJECT_SOURCE_DIR})
if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    set(CMAKE_INSTALL_PREFIX ${PROJECT_SOURCE_DIR})
    message("Set install folder to " ${CMAKE_INSTALL_PREFIX})
endif()

include_directories(${MIGRAPHX_FOLDER}/src/include
                    ${MIGRAPHX_FOLDER}/src/targets/gpu/include
                    ${MIGRAPHX_FOLDER}/src/targets/cpu/include
                    ${MIGRAPHX_FOLDER}/test/include
                    ${DEPS_FOLDER}/include)

link_directories(${MIGRAPHX_BUILD}/src/onnx 
                 ${MIGRAPHX_BUILD}/src
                 ${MIGRAPHX_BUILD}/src/targets/gpu
                 ${MIGRAPHX_BUILD}/src/targets/cpu
                 ${DEPS_FOLDER}/lib)


file(GLOB SRC_FILES "src/*.cpp")

foreach(filepath_name ${SRC_FILES})
    get_filename_component(src_name ${filepath_name} NAME)
    get_filename_component(bin_name ${src_name} NAME_WE)
    message("source file: " ${src_name} " ---> bin: " ${bin_name})
    add_executable(${bin_name} ${filepath_name})
    target_link_libraries(${bin_name} migraphx migraphx_cpu migraphx_device migraphx_gpu migraphx_onnx hip_hcc MIOpen)
endforeach(filepath_name)

