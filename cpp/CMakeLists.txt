cmake_minimum_required(VERSION 3.16)
project(PointCloudGeoLabCppDemo LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(kdtree_demo kdtree_demo.cpp)

if(MSVC)
  target_compile_options(kdtree_demo PRIVATE /W4)
else()
  target_compile_options(kdtree_demo PRIVATE -Wall -Wextra -Wpedantic)
endif()
