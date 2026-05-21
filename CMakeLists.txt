cmake_minimum_required(VERSION 3.10)
project(SISAP_Task1)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -ffast-math -Wall -Wextra -fopenmp")

find_package(OpenMP REQUIRED)
find_package(HDF5 REQUIRED COMPONENTS CXX)

add_executable(main src/main.cpp)

target_include_directories(main PRIVATE ${HDF5_INCLUDE_DIRS})
target_link_libraries(main PRIVATE OpenMP::OpenMP_CXX ${HDF5_CXX_LIBRARIES} ${HDF5_LIBRARIES})