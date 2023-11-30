cmake_minimum_required(VERSION 3.20)
project(Batch-Scheduler LANGUAGES CXX CUDA)

enable_language(CUDA)
find_package(CUDA REQUIRED)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};--gpu-architecture=sm_70)

include_directories(${CUDA_INCLUDE_DIRS})

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(SOURCE_FILES
    multi-shot.cu)


add_executable(multi-shot ${SOURCE_FILES})

target_link_libraries(multi-shot ${CUDA_LIBRARIES})