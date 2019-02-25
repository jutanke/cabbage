cmake_minimum_required(VERSION 2.8)
project( Cabbage )
find_package( OpenCV REQUIRED )
find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

INCLUDE_DIRECTORIES("external/graph/include")
set (CMAKE_CXX_STANDARD 11)

add_executable( sample src/cabbage.cpp )
target_link_libraries( sample ${OpenCV_LIBS} )
