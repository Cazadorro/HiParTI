cmake_minimum_required(VERSION 3.25)  # CMake 3.2 supports CUDA 7.0
project(spmv)
set(CMAKE_CXX_STANDARD 20)

add_library(spmv-csrk STATIC)

target_sources(spmv-csrk PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/csrk.cpp

)

target_sources(spmv-csrk PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/csrk.h
)

target_include_directories(spmv-csrk PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/./
)
set_property(TARGET spmv-csrk PROPERTY  MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
find_package(Boost REQUIRED COMPONENTS graph)
find_package(OpenMP)
target_link_libraries(spmv-csrk PUBLIC OpenMP::OpenMP_CXX Boost::graph)

add_executable(spmv-csrk-example spmv.cpp)
set_property(TARGET spmv-csrk-example PROPERTY  MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
target_link_libraries(spmv-csrk-example PRIVATE spmv-csrk)

