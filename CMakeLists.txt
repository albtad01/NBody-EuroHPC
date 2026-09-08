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
option (ENABLE_MURB_CUDA "Enable to compile the MUrB CUDA executable"  OFF)
option (ENABLE_MURB_MPI  "Enable the four-GPU MPI backend"             OFF)

if (NOT ENABLE_MURB)
    set (ENABLE_TEST OFF)
endif()
if (ENABLE_MURB_MPI AND NOT ENABLE_MURB_CUDA)
    message(FATAL_ERROR "ENABLE_MURB_MPI requires ENABLE_MURB_CUDA")
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
    find_package (OpenGL REQUIRED)
    find_package (GLEW REQUIRED)
    find_package (GLM REQUIRED)
    find_package (GLFW REQUIRED)
    set (LINK_VISU 1)
    message (STATUS "MUrB OpenGL visualization: ENABLED")
else ()
    message (STATUS "MUrB OpenGL visualization: DISABLED")
endif ()

# --- Part 1: Common Library ---
file (GLOB_RECURSE source_common src/common/*.cpp src/common/*.hpp)
if (ENABLE_MURB_CUDA)
    # Only CUDABodies is needed by the supported single-node CUDA backends.
    list (APPEND source_common src/common/core/CUDABodies.cu)
endif()

add_library (common-lib OBJECT ${source_common})
list(APPEND murb_targets_list common-lib)

# --- Part 2: MUrB Implementation ---
if (ENABLE_MURB)
    # Keep the supported single-node build independent of deferred MPI,
    # replay, tracking, leapfrog, heterogeneous, and experimental sources.
    set (source_murb_implem_files
        src/murb/implem/SimulationNBodyNaive.cpp
        src/murb/implem/SimulationNBodyOptim.cpp
        src/murb/implem/SimulationNBodySIMD.cpp
        src/murb/implem/SimulationNBodyOpenMP.cpp
        src/murb/implem/SimulationNBodyBinaryPlayer.cpp
    )
    if (ENABLE_MURB_CUDA)
        list (APPEND source_murb_implem_files
            src/murb/implem/SimulationNBodyCUDATile.cu
            src/murb/implem/SimulationNBodyCUDATileFullDevice.cu
            src/murb/implem/SimulationNBodyCUDATileFullDevice200k.cu
        )
        if (ENABLE_MURB_MPI)
            list (APPEND source_murb_implem_files
                src/murb/implem/MultiGpuRuntime.cpp
                src/murb/implem/SimulationNBodyMultiNodeCUDA.cu
            )
        endif()
    endif()

    add_library (murb-implem-lib OBJECT ${source_murb_implem_files})
    list(APPEND murb_targets_list murb-implem-lib)

    file (GLOB_RECURSE source_murb_main_file src/murb/main.cpp)
    add_executable (murb-bin $<TARGET_OBJECTS:common-lib> $<TARGET_OBJECTS:murb-implem-lib> ${source_murb_main_file})
    set_target_properties (murb-bin PROPERTIES OUTPUT_NAME murb)
    list(APPEND murb_targets_list murb-bin)
    target_compile_definitions(murb-bin PUBLIC NBODY_FLOAT)

    if (ENABLE_TEST)
        # CUDA backend correctness is exercised by the supported C++ tests.
        # Deferred GPU history/replay tests remain out of this target.
        file (GLOB_RECURSE source_test_files src/test/*.cpp src/test/*.hpp)
        add_executable (test-bin $<TARGET_OBJECTS:common-lib> $<TARGET_OBJECTS:murb-implem-lib> ${source_test_files})
        set_target_properties (test-bin PROPERTIES OUTPUT_NAME murb-test)
        list(APPEND murb_targets_list test-bin)
        target_include_directories (test-bin PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/lib/Catch2/include/")
        target_include_directories (test-bin PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src/murb/implem")
        enable_testing()
        add_test(NAME murb::test COMMAND test-bin)
        add_test(NAME murb::trajectory-cli
            COMMAND ${CMAKE_COMMAND}
                -DMURB_EXECUTABLE=$<TARGET_FILE:murb-bin>
                -DTEST_DIRECTORY=${CMAKE_CURRENT_BINARY_DIR}/trajectory-cli-test
                -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/TestTrajectoryCLI.cmake)
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

# Keep MPI opt-in so CPU, macOS, and validated single-A100 builds remain
# independent. This phase supports exactly four ranks on one Booster node.
if (ENABLE_MURB_MPI AND ENABLE_MURB)
    find_package(MPI REQUIRED COMPONENTS CXX)
    target_compile_definitions(murb-implem-lib PRIVATE USE_MPI)
    target_compile_definitions(murb-bin PRIVATE USE_MPI)
    target_link_libraries(murb-implem-lib PRIVATE MPI::MPI_CXX)
    target_link_libraries(murb-bin PRIVATE MPI::MPI_CXX)
    if (ENABLE_TEST)
        target_compile_definitions(test-bin PRIVATE USE_MPI)
        target_link_libraries(test-bin PRIVATE MPI::MPI_CXX)
    endif()
    message(STATUS "MUrB four-GPU MPI backend: ENABLED (explicit host staging)")
else()
    message(STATUS "MUrB four-GPU MPI backend: DISABLED")
endif()

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
    targets_compile_definitions("${murb_targets_list}" PRIVATE
        MURB_SHADER_DIR="${CMAKE_CURRENT_SOURCE_DIR}/src/common/ogl/shaders")
endif ()

# OpenMP support
if (ENABLE_MURB_OMP)
    find_package (OpenMP REQUIRED)
    if (OpenMP_FOUND)
        targets_link_libraries("${murb_targets_list}" PUBLIC OpenMP::OpenMP_CXX)
    endif ()
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
