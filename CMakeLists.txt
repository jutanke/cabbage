cmake_minimum_required(VERSION 2.8)
project( DisplayImage )
find_package( OpenCV REQUIRED )
find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

find_package(Boost 1.58.0 REQUIRED)

set (CMAKE_CXX_STANDARD 11)

add_executable( Sample src/main.cpp )
target_link_libraries( Sample ${OpenCV_LIBS} )

add_library(cabbage src/cabbage.cpp)
