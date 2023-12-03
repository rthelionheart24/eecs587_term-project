cmake_minimum_required(VERSION 3.20)
project(Batch-Scheduler LANGUAGES CXX CUDA)

enable_language(CUDA)
find_package(CUDA REQUIRED)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode arch=compute_89,code=sm_89)

include_directories(${CUDA_INCLUDE_DIRS})
link_directories(${CUDA_LIBRARY_DIRS})

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(SOURCE_FILES
    multi-shot.cu)

set(CMAKE_CUDA_FLAGS_DEBUG "-G -g")


add_executable(multi-shot ${SOURCE_FILES})

target_link_libraries(multi-shot ${CUDA_LIBRARIES})