cmake_minimum_required(VERSION 3.18)
project(dms_trt_engine LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# ─── TensorRT ───
find_path(TENSORRT_INCLUDE_DIR NvInfer.h
    HINTS /usr/include/x86_64-linux-gnu /usr/local/TensorRT/include
          $ENV{TENSORRT_ROOT}/include)
find_library(TENSORRT_LIB nvinfer
    HINTS /usr/lib/x86_64-linux-gnu /usr/local/TensorRT/lib
          $ENV{TENSORRT_ROOT}/lib)
find_library(TENSORRT_PLUGIN nvparsers
    HINTS /usr/lib/x86_64-linux-gnu /usr/local/TensorRT/lib
          $ENV{TENSORRT_ROOT}/lib)

# ─── CUDA ───
find_package(CUDA REQUIRED)

# ─── OpenCV ───
find_package(OpenCV REQUIRED)

message(STATUS "TensorRT include: ${TENSORRT_INCLUDE_DIR}")
message(STATUS "TensorRT lib:     ${TENSORRT_LIB}")
message(STATUS "OpenCV:           ${OpenCV_VERSION}")

# ─── Build ───
add_executable(dms_engine
    src/main.cpp
    src/dms_engine.cpp
    src/postprocess.cpp
    src/preprocess.cu
)

target_include_directories(dms_engine PRIVATE
    include
    ${TENSORRT_INCLUDE_DIR}
    ${CUDA_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(dms_engine PRIVATE
    ${TENSORRT_LIB}
    ${CUDA_LIBRARIES}
    ${OpenCV_LIBS}
    nvinfer
    nvonnxparser
    cudart
)
