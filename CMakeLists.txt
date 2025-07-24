cmake_minimum_required(VERSION 3.17)  # CMake version check
project(ecc_gpu)
set(CMAKE_CXX_STANDARD 14)

find_package(OpenCV REQUIRED)
find_package(CUDAToolkit REQUIRED)
find_package(Python COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 CONFIG REQUIRED)
find_package(CUDA REQUIRED)

include_directories( ${OpenCV_INCLUDE_DIRS} )
include_directories( ${CUDAToolkit_INCLUDE_DIRS} )

include_directories(ecc)

set(SOURCE_FILES ecc/ecc_cuda.cpp ecc/ecc_cuda.h ecc/python_binding.cpp )

pybind11_add_module(ecc_gpu ${SOURCE_FILES})
target_link_libraries(ecc_gpu PRIVATE nppist ${OpenCV_LIBS} ${CUDA_LIBRARY} ${CUDA_CUDA_LIBRARY} ${CUDA_cudart_static_LIBRARY})     