cmake_minimum_required(VERSION 3.25)  # CMake 3.2 supports CUDA 7.0
project(HiParTI)



if(USE_CUDA)
    file(GLOB_RECURSE BENCH_SRCS RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "*.c" "*.cpp" "*.cu")
else()
    file(GLOB_RECURSE BENCH_SRCS RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "*.c" "*.cpp")
endif()

#find_package(Tracy CONFIG REQUIRED)
#include(FetchContent)
#FetchContent_Declare(
#        tracy
#        GIT_REPOSITORY https://github.com/wolfpld/tracy.git
#        GIT_TAG master
#        GIT_SHALLOW TRUE
#        GIT_PROGRESS TRUE
#)
#FetchContent_MakeAvailable(tracy)

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
        set_property(TARGET "${BENCH_EXE}" PROPERTY  MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")

    else()
        target_link_libraries("${BENCH_EXE}" PRIVATE HiParTI )
#        set_property(TARGET "${BENCH_EXE}" PROPERTY  MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:DebugDLL>")
    endif()
#    target_link_libraries("${BENCH_EXE}" PUBLIC Tracy::TracyClient)

#    target_link_libraries("${BENCH_EXE}" PUBLIC TracyClient)
    target_compile_definitions("${BENCH_EXE}" PRIVATE -DTRACEY_ENABLE)
    if(USE_CUDA)
    endif()
endforeach(BENCH_SRC)
