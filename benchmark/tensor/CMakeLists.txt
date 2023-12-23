cmake_minimum_required(VERSION 3.2)  # CMake 3.2 supports CUDA 7.0
project(HiParTI)



if(USE_CUDA)
    file(GLOB_RECURSE BENCH_SRCS RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "*.c" "*.cpp" "*.cu")
else()
    file(GLOB_RECURSE BENCH_SRCS RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "*.c" "*.cpp")
endif()


foreach(BENCH_SRC IN LISTS BENCH_SRCS)
    get_filename_component(BENCH_EXE "${BENCH_SRC}" NAME_WE)
#    if(USE_CUDA)
#        cuda_add_executable("${BENCH_EXE}" "${BENCH_SRC}")
#    else()
#        add_executable("${BENCH_EXE}" "${BENCH_SRC}")
#    endif()
    add_executable("${BENCH_EXE}" "${BENCH_SRC}")
    target_sources("${BENCH_EXE}" PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/hicoo_utils.h
            ${CMAKE_CURRENT_SOURCE_DIR}/hicoo_utils.cpp)
    target_include_directories("${BENCH_EXE}" PRIVATE  ${CMAKE_CURRENT_SOURCE_DIR}/)
#    set_target_properties("${BENCH_EXE}" PROPERTIES C_STANDARD 99)
    if(BUILD_STATIC)
        target_link_libraries("${BENCH_EXE}" PRIVATE HiParTI_s)
    else()
        target_link_libraries("${BENCH_EXE}" PRIVATE HiParTI)
    endif()
    if(USE_CUDA)
    endif()
endforeach(BENCH_SRC)
