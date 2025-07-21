cmake_minimum_required(VERSION 3.17)  # CMake version check
project(ecc_gpu)               
set(CMAKE_CXX_STANDARD 14)         
find_package(OpenCV REQUIRED )
find_package(CUDAToolkit REQUIRED)
find_package(Python COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 CONFIG REQUIRED)
find_package(CUDA REQUIRED)


include_directories( ${OpenCV_INCLUDE_DIRS} )
include_directories( ${CUDAToolkit_INCLUDE_DIRS} )

add_executable(ecc_gpu ecc/main.cpp ecc/ecc_cuda.cpp)

target_link_libraries(ecc_gpu nppist cudart ${OpenCV_LIBS})

pybind11_add_module(ecc_gpu ${SOURCE_FILES})
TARGET_LINK_LIBRARIES (ecc_gpu PRIVATE ${OpenCV_LIBS} ${CUDA_LIBRARY} ${CUDA_CUDA_LIBRARY} ${CUDA_cudart_static_LIBRARY})         
