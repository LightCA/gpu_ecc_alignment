cmake_minimum_required(VERSION 3.17)  # CMake version check
project(ecc_gpu) 
    
#set(CMAKE_BUILD_TYPE Debug) # for vsCode debug
set(CMAKE_CXX_STANDARD 14) 
  
find_package(OpenCV 4.5 REQUIRED )
find_package(CUDAToolkit REQUIRED)

include_directories( ${OpenCV_INCLUDE_DIRS} )
include_directories( ${CUDAToolkit_INCLUDE_DIRS} )
include_directories( ecc )

add_executable(ecc_gpu ecc/main.cpp ecc/ecc_cuda.cpp)

target_link_libraries(ecc_gpu nppist cudart ${OpenCV_LIBS} )