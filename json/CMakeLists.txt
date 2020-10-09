cmake_minimum_required(VERSION 3.13.4)
project (Test_json)

#find_package(nlohmann_json REQUIRED)

message("PACKAGE_FIND_VERSION_MAJOR = " ${PACKAGE_FIND_VERSION_MAJOR})
message("nlohmann_FOUND = " ${nlohmann_FOUND})
message("CMAKE_FIND_PACKAGE_NAME = " ${CMAKE_FIND_PACKAGE_NAME})
message("cmake_current_src_dir = " ${CMAKE_SOURCE_DIR})
message("cmake_current_bin_dir = " ${CMAKE_BINARY_DIR})

message("CMAKE_PREFIX_PATH = " ${CMAKE_PREFIX_PATH})
include_directories(${CMAKE_PREFIX_PATH}/include)
set (CMAKE_CXX_STANDARD 14)
#add_executable(test_json1 test_json1)

file(GLOB test_examples test_*.cpp)

foreach(filepath_name ${test_examples})
    get_filename_component(src_name ${filepath_name} NAME)
    get_filename_component(bin_name ${src_name} NAME_WE)
    message("source file: " ${src_name} " ---> bin: " ${bin_name})
    add_executable(${bin_name} ${src_name})
endforeach(filepath_name)

