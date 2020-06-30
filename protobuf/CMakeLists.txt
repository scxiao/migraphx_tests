cmake_minimum_required(VERSION 3.5)
project (Test_protobuf)


set (CMAKE_PREFIX_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cget)

find_package(Protobuf REQUIRED)

set (CMAKE_CXX_STANDARD 14)
set (CMAKE_VERBOSE_MAKEFILE ON)


protobuf_generate_cpp(PROTO_SRCS PROTO_HDRS onnx.proto)
add_library(onnx-proto STATIC ${PROTO_SRCS})
target_include_directories(onnx-proto SYSTEM PUBLIC ${CMAKE_CURRENT_BINARY_DIR} ${PROTOBUF_INCLUDE_DIR})
target_compile_options(onnx-proto PRIVATE -w)
target_link_libraries(onnx-proto PRIVATE ${PROTOBUF_LIBRARY})
set_target_properties(onnx-proto PROPERTIES POSITION_INDEPENDENT_CODE On)

set(test_examples parse_onnx.cpp)

foreach(filepath_name ${test_examples})
    get_filename_component(src_name ${filepath_name} NAME)
    get_filename_component(bin_name ${src_name} NAME_WE)
    message("source file: " ${src_name} " ---> bin: " ${bin_name})
    add_executable(${bin_name} ${src_name})
    target_link_libraries(${bin_name} onnx-proto)
#    add_dependencies(${bin_name} copylibs)
endforeach(filepath_name)

