cmake_minimum_required(VERSION 3.5)
project (Test_protobuf)

set (MIGRAPHX_FOLDER /home/scxiao/Workplace/projects/AMDMIGraphX)
#set (MIGRAPHX_FOLDER /home/scxiao/Workplace/projects/MIGraph)
#set (MIGRAPHX_FOLDER /home/scxiao/Workplace/projects/migraphx_onnx_backend)
set (MIGRAPHX_BUILD ${MIGRAPHX_FOLDER}/build_debug)
#set (MIGRAPHX_BUILD ${MIGRAPHX_FOLDER}/build)


set (CMAKE_PREFIX_PATH /home/scxiao/Workplace/projects/MIGraph/deps_json)
find_package(Protobuf REQUIRED)

set (CMAKE_CXX_STANDARD 17)

include_directories(${MIGRAPHX_FOLDER}/src/api/include
                    ${MIGRAPHX_FOLDER}/src/include)
link_directories(${MIGRAPHX_BUILD}/lib)

protobuf_generate_cpp(PROTO_SRCS PROTO_HDRS onnx.proto)
add_library(onnx-proto STATIC ${PROTO_SRCS})
target_include_directories(onnx-proto SYSTEM PUBLIC ${CMAKE_CURRENT_BINARY_DIR} ${PROTOBUF_INCLUDE_DIR})
target_compile_options(onnx-proto PRIVATE -w)
target_link_libraries(onnx-proto PRIVATE ${PROTOBUF_LIBRARY})
#set_target_properties(onnx-proto PROPERTIES POSITION_INDEPENDENT_CODE On)

set(test_examples parse_onnx.cpp parse_tensor.cpp)

foreach(filepath_name ${test_examples})
    get_filename_component(src_name ${filepath_name} NAME)
    get_filename_component(bin_name ${src_name} NAME_WE)
    message("source file: " ${src_name} " ---> bin: " ${bin_name})
    add_executable(${bin_name} ${src_name})
    target_link_libraries(${bin_name} onnx-proto migraphx_c)
#    add_dependencies(${bin_name} copylibs)
endforeach(filepath_name)

