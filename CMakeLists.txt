cmake_minimum_required(VERSION 3.17)  # CMake version check
project(ecc_gpu) 
    
set(CMAKE_CXX_STANDARD 14) 
  
set(OpenCV_DIR "C:/Users/hadar/Documents/opencvBuilds/build/install")
find_package(OpenCV 4.5 REQUIRED )
find_package(CUDAToolkit REQUIRED)

include_directories( ${CUDAToolkit_INCLUDE_DIRS} )

add_executable(ecc_gpu ecc/main.cpp ecc/ecc_cuda.cpp)

target_link_libraries(ecc_gpu nppist cudart ${OpenCV_LIBS})

#   cmake --build . --config Release  #     For Multi-Configuration Generators (e.g., Visual Studio) usually on windows