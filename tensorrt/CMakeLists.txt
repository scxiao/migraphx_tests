cmake_minimum_required(VERSION 3.5)
project (Test_Tensorrt)

set (MIGRAPHX_FOLDER /home/scxiao/Workplace/projects/AMDMIGraphX)
set (DEPS_FOLDER ${MIGRAPHX_FOLDER}/deps_rocblas2.7)
#set (MIGRAPHX_BUILD  ${MIGRAPHX_FOLDER}/build_debug)
set (MIGRAPHX_BUILD  ${MIGRAPHX_FOLDER}/build)
#set (CMAKE_CXX_COMPILER /opt/rocm/bin/hcc)
set (CMAKE_CXX_STANDARD 14)

add_definitions(-DBLIBCXX_DEBUG)
message("current folder is: " ${PROJECT_SOURCE_DIR})
if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    set(CMAKE_INSTALL_PREFIX ${PROJECT_SOURCE_DIR})
    message("Set install folder to " ${CMAKE_INSTALL_PREFIX})
endif()

#find_package(opencv REQUIRED /user/local/lib)

include_directories(/workspace/TensorRT/include
                    /usr/local/include/opencv4
                    /usr/local/cuda/include)
link_directories(/tensorrt/lib
                 /usr/local/cuda/lib64
                 /usr/local/lib)
add_executable(test test_load_onnx.cpp)
target_link_libraries(test nvonnxparser nvinfer cudart opencv_cudaarithm opencv_cudabgsegm opencv_cudawarping opencv_core opencv_cudabgsegm opencv_cudaarithm opencv_imgcodecs)

#file(GLOB SRC_FILES "src/*.cpp")
#foreach(filepath_name ${SRC_FILES})
#    get_filename_component(src_name ${filepath_name} NAME)
#    get_filename_component(bin_name ${src_name} NAME_WE)
#    message("source file: " ${src_name} " ---> bin: " ${bin_name})
#    add_executable(${bin_name} ${filepath_name})
#    target_link_libraries(${bin_name} migraphx migraphx_cpu migraphx_device migraphx_gpu migraphx_onnx hip_hcc MIOpen)
#endforeach(filepath_name)

