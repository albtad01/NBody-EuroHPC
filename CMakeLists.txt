# CMake entry point
cmake_minimum_required (VERSION 3.21)
# Invalidate a previously successful build even if this configure later fails.
file(REMOVE "${CMAKE_BINARY_DIR}/murb-build.ready")
project (murb CXX)

# --- GLOBAL STANDARDS (Required for <concepts> and performance) ---
set (CMAKE_CXX_STANDARD 20)
set (CMAKE_CXX_STANDARD_REQUIRED ON)
set (CMAKE_CUDA_STANDARD 17) 
set (CMAKE_CUDA_STANDARD_REQUIRED ON)

# Command line options
option (ENABLE_MURB      "Enable to compile the MUrB executable"        ON )
option (ENABLE_VISU      "Enable the OpenGL visualization"              ON )
option (ENABLE_TEST      "Enable test program to validate MUrB kernels" ON )
option (ENABLE_MURB_OMP  "Enable to compile the MUrB OMP executable"    ON )
option (ENABLE_MURB_OCL  "Enable to compile the MUrB OCL executable"    OFF)
option (ENABLE_MURB_CUDA "Enable to compile MUsB CUDA executable"       OFF)

if (NOT ENABLE_MURB)
    set (ENABLE_TEST OFF)
endif()

# Paths and Modules setup
set (CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules/")
set (EXECUTABLE_OUTPUT_PATH bin/)

# Enable CUDA if requested
if (ENABLE_MURB_CUDA)
    include(CheckLanguage)
    check_language(CUDA)
    enable_language(CUDA)
endif ()

# Find visualization dependencies
set(LINK_VISU 0)
if (ENABLE_VISU)
    set (OpenGL_GL_PREFERENCE LEGACY)
    find_package (OpenGL)
    find_package (GLEW)
    find_package (GLM)
    find_package (GLFW)
    if ((OPENGL_FOUND) AND (GLEW_FOUND) AND (GLM_FOUND) AND (GLFW_FOUND))
        set (LINK_VISU 1)
    else ()
        set (LINK_VISU 0)
    endif ()
endif ()

# --- Part 1: Common Library ---
file (GLOB_RECURSE source_common src/common/*.cpp src/common/*.hpp)
if (ENABLE_MURB_CUDA)
    file (GLOB_RECURSE source_common_cuda src/common/*.cu)
    list (APPEND source_common ${source_common_cuda})
endif()

add_library (common-lib OBJECT ${source_common})
list(APPEND murb_targets_list common-lib)

# --- Part 2: MUrB Implementation ---
if (ENABLE_MURB)
    file (GLOB_RECURSE source_murb_implem_files src/murb/implem/*.cpp src/murb/implem/*.hpp)
    if (ENABLE_MURB_CUDA)
        file (GLOB_RECURSE source_cuda_files src/murb/implem/*.cu)
        list (APPEND source_murb_implem_files ${source_cuda_files})
    endif()

    add_library (murb-implem-lib OBJECT ${source_murb_implem_files})
    list(APPEND murb_targets_list murb-implem-lib)

    file (GLOB_RECURSE source_murb_main_file src/murb/main.cpp)
    add_executable (murb-bin $<TARGET_OBJECTS:common-lib> $<TARGET_OBJECTS:murb-implem-lib> ${source_murb_main_file})
    set_target_properties (murb-bin PROPERTIES OUTPUT_NAME murb)
    list(APPEND murb_targets_list murb-bin)
    target_compile_definitions(murb-bin PUBLIC NBODY_FLOAT)

    if (ENABLE_TEST)
        file (GLOB_RECURSE source_test_files src/test/*.cpp src/test/*.hpp)
        if (ENABLE_MURB_CUDA)
            file (GLOB_RECURSE cuda_test_files src/test/*.cu)
            list (APPEND source_test_files ${cuda_test_files})
        endif()
        add_executable (test-bin $<TARGET_OBJECTS:common-lib> $<TARGET_OBJECTS:murb-implem-lib> ${source_test_files})
        set_target_properties (test-bin PROPERTIES OUTPUT_NAME murb-test)
        list(APPEND murb_targets_list test-bin)
        target_include_directories (test-bin PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/lib/Catch2/include/")
        target_include_directories (test-bin PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src/murb/implem")
        enable_testing()
        add_test(NAME murb::test COMMAND test-bin)
    endif ()
endif ()

# --- Part 3: Macros and Optimizations ---
macro (targets_compile_definitions targets privacy def)
    foreach(_target IN ITEMS ${targets})
        target_compile_definitions (${_target} ${privacy} ${def})
    endforeach()
endmacro()

macro (targets_include_directories targets privacy dir)
    foreach(_target IN ITEMS ${targets})
        target_include_directories (${_target} ${privacy} ${dir})
    endforeach()
endmacro()

macro (targets_link_libraries targets privacy lib)
    foreach(_target IN ITEMS ${targets})
        target_link_libraries (${_target} ${privacy} ${lib})
    endforeach()
endmacro()

macro (targets_compile_options targets privacy)
    foreach(_target IN ITEMS ${targets})
        target_compile_options(${_target} ${privacy} ${ARGN})
    endforeach()
endmacro()

# Global config and Includes
targets_include_directories ("${murb_targets_list}" PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src/common")

if (EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/lib/MIPP/src/mipp.h")
    targets_compile_definitions ("${murb_targets_list}" PRIVATE MIPP_ENABLE_BACKTRACE)
    targets_include_directories ("${murb_targets_list}" PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/lib/MIPP/src/")
else ()
    message(FATAL_ERROR "MIPP missing.")
endif ()

# --- CRITICAL OPTIMIZATIONS ---
# Preserve the effective arithmetic flags of the audited baseline: host -O3,
# CUDA --use_fast_math. The old helper silently discarded later arguments.
targets_compile_options("${murb_targets_list}" PRIVATE -O3)

# CUDA Optimizations
if (ENABLE_MURB_CUDA AND ENABLE_MURB)
    find_package(CUDAToolkit REQUIRED)
    targets_compile_definitions("${murb_targets_list}" PRIVATE USE_CUDA)
    targets_link_libraries("${murb_targets_list}" PRIVATE CUDA::cudart)
    targets_compile_options("${murb_targets_list}" PRIVATE 
        $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
    )
endif ()

# Visualization linking
if (LINK_VISU)
    if (OPENGL_FOUND)
        targets_link_libraries ("${murb_targets_list}" PRIVATE OpenGL::GL)
    endif ()
    if (GLEW_FOUND)
        targets_include_directories ("${murb_targets_list}" PRIVATE ${GLEW_INCLUDE_DIRS})
        targets_link_libraries ("${murb_targets_list}" PRIVATE GLEW::GLEW)
    endif ()
    if (GLM_FOUND)
        targets_include_directories ("${murb_targets_list}" PRIVATE ${GLM_INCLUDE_DIRS})
    endif ()
    if (GLFW_FOUND)
        targets_include_directories ("${murb_targets_list}" PRIVATE ${GLFW_INCLUDE_DIR})
        targets_link_libraries ("${murb_targets_list}" PRIVATE "${GLFW_LIBRARIES}")
    endif ()
    targets_compile_definitions("${murb_targets_list}" PRIVATE VISU)
endif ()

# OpenMP support
if (ENABLE_MURB_OMP)
    find_package (OpenMP REQUIRED)
    if (OpenMP_FOUND)
        targets_link_libraries("${murb_targets_list}" PUBLIC OpenMP::OpenMP_CXX)
    endif ()
endif()

# MPI setup
find_package(MPI REQUIRED COMPONENTS CXX)
if (ENABLE_MURB)
    # MPI implementations remain compiled, though unsupported by the Phase 1 CLI.
    target_link_libraries(murb-implem-lib PUBLIC MPI::MPI_CXX)
    target_link_libraries(murb-bin PRIVATE MPI::MPI_CXX)
    if (ENABLE_TEST)
        target_link_libraries(test-bin PRIVATE MPI::MPI_CXX)
    endif()
endif()

# OpenCL support
if (ENABLE_MURB_OCL AND ENABLE_MURB)
    targets_compile_definitions("${murb_targets_list}" PRIVATE USE_OCL)
    find_package (OpenCL REQUIRED)
    if (OpenCL_FOUND)
        targets_link_libraries("${murb_targets_list}" PUBLIC OpenCL::OpenCL)
    endif ()
endif ()
# Record the source identity actually compiled, and certify only a completed build.
if (ENABLE_MURB)
    add_custom_target(murb-build-info
        COMMAND ${CMAKE_COMMAND}
            -DSOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}
            -DOUTPUT_DIR=${CMAKE_CURRENT_BINARY_DIR}/generated
            -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/WriteBuildInfo.cmake
        BYPRODUCTS ${CMAKE_CURRENT_BINARY_DIR}/generated/BuildInfo.hpp
        VERBATIM)
    add_dependencies(murb-bin murb-build-info)
    target_include_directories(murb-bin PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/generated)
    set(ready_targets murb-bin)
    if (ENABLE_TEST)
        add_dependencies(test-bin murb-build-info)
        target_include_directories(test-bin PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/generated)
        list(APPEND ready_targets test-bin)
    endif()
    add_custom_target(murb-build-ready ALL
        COMMAND ${CMAKE_COMMAND} -E touch ${CMAKE_CURRENT_BINARY_DIR}/murb-build.ready
        DEPENDS ${ready_targets}
        VERBATIM)
endif()
