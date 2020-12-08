cmake_minimum_required(VERSION 3.5)
project (Tensorrt_mrpc)

add_definitions(-DBLIBCXX_DEBUG)
message("current folder is: " ${PROJECT_SOURCE_DIR})
if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    set(CMAKE_INSTALL_PREFIX ${PROJECT_SOURCE_DIR})
    message("Set install folder to " ${CMAKE_INSTALL_PREFIX})
endif()

include_directories(/workspace/TensorRT/include
                    /usr/local/include/opencv4
                    /usr/local/cuda/include)
link_directories(/tensorrt/lib
                 /usr/local/cuda/lib64
                 /usr/local/lib)
add_executable(test tensorrt_mrpc.cpp)
target_link_libraries(test nvonnxparser nvinfer cudart)

