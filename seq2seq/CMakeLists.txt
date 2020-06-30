cmake_minimum_required(VERSION 3.5)
project (Seq2seq_Translation)

set (MIGRAPHX_FOLDER /home/scxiao/Workplace/projects/MIGraph)
#set (MIGRAPHX_BUILD  ${MIGRAPHX_FOLDER}/build_debug)
set (MIGRAPHX_BUILD  ${MIGRAPHX_FOLDER}/build)
set (MIGRAPHX_DEPS   /home/scxiao/Workplace/projects/AMDMIGraphX/deps_rocm3.3)
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
                    ${MIGRAPHX_DEPS}/include
                    inc)

link_directories(${MIGRAPHX_BUILD}/lib
                 ${MIGRAPHX_DEPS}/lib
                 /opt/rocm/lib)


file(GLOB SRC_FILES "src/*.cpp")
set(bin_name seq2seq_trans)
add_executable(${bin_name} ${SRC_FILES})
target_link_libraries(${bin_name} migraphx migraphx_cpu migraphx_device migraphx_gpu migraphx_onnx hip_hcc MIOpen)

