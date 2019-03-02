cmake_minimum_required(VERSION 3.5)
project (Seq2seq_Translation)

set (MIGRAPHX_FOLDER /home/scxiao/Workplace/projects/AMDMIGraphX)
set (MIGRAPHX_BUILD  ${MIGRAPHX_FOLDER}/build)
set (MIGRAPHX_DEPS   ${MIGRAPHX_FOLDER}/deps_py/lib)
#set (MIGRAPHX_LIB_PATH /home/scxiao/Workplace/software/migraphlibs)
set (CMAKE_CXX_COMPILER /opt/rocm/bin/hcc)
set (CMAKE_CXX_STANDARD 14)
set (DEBUG _debug)

#add_custom_target(
#   copylibs
#	COMMAND cp ${MIGRAPHX_BUILD}/src/onnx/libmigraphx_onnx.so          ${MIGRAPHX_LIB_PATH}/.
#	COMMAND cp ${MIGRAPHX_BUILD}/src/libmigraphx.so                    ${MIGRAPHX_LIB_PATH}/.
#	COMMAND cp ${MIGRAPHX_BUILD}/src/targets/cpu/libmigraphx_cpu.so    ${MIGRAPHX_LIB_PATH}/.
#	COMMAND cp ${MIGRAPHX_BUILD}/src/targets/gpu/libmigraphx_gpu.so    ${MIGRAPHX_LIB_PATH}/.
#	COMMAND cp ${MIGRAPHX_BUILD}/src/targets/gpu/libmigraphx_device.so ${MIGRAPHX_LIB_PATH}/.)


include_directories(${MIGRAPHX_FOLDER}/src/include
                    ${MIGRAPHX_FOLDER}/src/targets/gpu/include
                    ${MIGRAPHX_FOLDER}/src/targets/cpu/include
                    ${MIGRAPHX_FOLDER}/test/include
                    ${MIGRAPHX_FOLDER}/deps_py/include
                    inc)

link_directories(${MIGRAPHX_BUILD}/src/onnx 
                 ${MIGRAPHX_BUILD}/src
                 ${MIGRAPHX_BUILD}/src/targets/gpu
                 ${MIGRAPHX_BUILD}/src/targets/cpu
                 ${MIGRAPHX_DEPS})

file(GLOB SRC_FILES "src/*.cpp")
set(bin_name seq2seq_trans)
add_executable(${bin_name} ${SRC_FILES})
target_link_libraries(${bin_name} migraphx migraphx_cpu migraphx_device migraphx_gpu migraphx_onnx hip_hcc MIOpen)
#   add_dependencies(${bin_name} copylibs)

