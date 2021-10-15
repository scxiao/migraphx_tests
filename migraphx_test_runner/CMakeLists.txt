cmake_minimum_required(VERSION 3.5)
project (migraphx_test_runner)

set(CMAKE_VERBOSE_MAKEFILE on)

#set(PROTOBUF_LIB protobuf::libprotobuf)
#add_subdirectory(${PROJECT_SOURCE_DIR}/protobuf/cmake EXCLUDE_FROM_ALL)
#set_target_properties(libprotoc PROPERTIES FOLDER "protobuf")
#add_executable(protobuf::protoc ALIAS protoc)
#
#set_target_properties(protoc PROPERTIES FOLDER "protobuf")
#set(PROTOBUF_LIB libprotobuf)
#install(TARGETS ${PROTOBUF_LIB}
#        PUBLIC_HEADER
#        ARCHIVE  DESTINATION ${CMAKE_INSTALL_LIBDIR}
#        LIBRARY  DESTINATION ${CMAKE_INSTALL_LIBDIR}
#        RUNTIME  DESTINATION ${CMAKE_INSTALL_BINDIR})


add_definitions(-D__HIP_PLATFORM_HCC__)

#set(DEFAULT_PROTOBUF_VERSION 3.11.0)
#set(PROTOBUF_VERSION ${DEFAULT_PROTOBUF_VERSION})
#message(STATUS "Protobuf version set to ${PROTOBUF_VERSION}")
#include(protobuf.cmake)
#configure_protobuf(${PROTOBUF_VERSION})

set (CMAKE_CXX_STANDARD 17)

#set(PROTO_PATH ${CMAKE_BINARY_DIR}/third_party.protobuf)
#list(APPEND CMAKE_PREFIX_PATH /home/scxiao/Workplace/projects/MIGraph/deps_json)

find_package(migraphx REQUIRED)
#find_package(Protobuf REQUIRED PATHS ${PROTO_PATH})
find_package(Protobuf REQUIRED)


include_directories(/opt/rocm-4.3.1/include)
protobuf_generate_cpp(PROTO_SRCS PROTO_HDRS onnx.proto)
add_library(onnx-proto STATIC ${PROTO_SRCS})
target_include_directories(onnx-proto SYSTEM PUBLIC ${CMAKE_CURRENT_BINARY_DIR} ${PROTOBUF_INCLUDE_DIR})
target_compile_options(onnx-proto PRIVATE -w)
target_link_libraries(onnx-proto PRIVATE ${PROTOBUF_LIBRARY})

file(GLOB src_files "*.cpp")
set (bin_name migraphx_test_runner)
message("source file: " ${src_name} " ---> bin: " ${bin_name})
add_executable(${bin_name} ${src_files})
target_link_libraries(${bin_name} migraphx::c onnx-proto stdc++fs ${PROTOBUF_LIBRARY})
