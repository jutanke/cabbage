cmake_minimum_required(VERSION 2.8)
project( SolveGraph )
find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

INCLUDE_DIRECTORIES(../external/graph/include)

set(Python_ADDITIONAL_VERSIONS 3.5)
find_package(Boost 1.63.0 COMPONENTS system thread graph REQUIRED)
SET(Boost_USE_STATIC_LIBS OFF)
SET(Boost_USE_MULTITHREADED ON)
SET(Boost_USE_STATIC_RUNTIME OFF)

set (CMAKE_CXX_STANDARD 11)

add_executable( Solver src/solve_graph.cpp )
