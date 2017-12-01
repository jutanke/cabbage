cmake_minimum_required(VERSION 2.8)
project( SolveGraph )
find_package( OpenCV REQUIRED )
find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

set(Python_ADDITIONAL_VERSIONS 3.5)
find_package(Boost 1.58.0 COMPONENTS system thread python-py35 graph REQUIRED)
SET(Boost_USE_STATIC_LIBS OFF)
SET(Boost_USE_MULTITHREADED ON)
SET(Boost_USE_STATIC_RUNTIME OFF)

set (CMAKE_CXX_STANDARD 11)

add_executable( Sample src/solve_graph.cpp )
target_link_libraries( SolveGraph ${OpenCV_LIBS} )
