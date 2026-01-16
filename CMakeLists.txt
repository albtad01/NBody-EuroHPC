# CMake entry point
cmake_minimum_required (VERSION 3.10.3)

project (murb CXX)

# Command line options
option (ENABLE_MURB      "Enable to compile the MUrB executable"        ON )
option (ENABLE_VISU      "Enable the OpenGL visualization"              ON )
option (ENABLE_TEST      "Enable test program to validate MUrB kernels" ON )
option (ENABLE_MURB_OMP  "Enable to compile the MUrB OMP executable"    ON )
option (ENABLE_MURB_OCL  "Enable to compile the MUrB OCL executable"    OFF)
option (ENABLE_MURB_CUDA "Enable to compile MUsB CUDA executable"       OFF)

if (NOT ENABLE_MURB)
    message("ENABLE_TEST has been switched OFF because ENABLE_MURB is disabled.")
    set (ENABLE_TEST OFF)
endif()

message(STATUS "MUrB options: ")
message(STATUS "  * ENABLE_MURB: '${ENABLE_MURB}'")
message(STATUS "  * ENABLE_VISU: '${ENABLE_VISU}'")
message(STATUS "  * ENABLE_TEST: '${ENABLE_TEST}'")
message(STATUS "  * ENABLE_MURB_OMP: '${ENABLE_MURB_OMP}'")
message(STATUS "  * ENABLE_MURB_OCL: '${ENABLE_MURB_OCL}'")
message(STATUS "  * ENABLE_MURB_CUDA: '${ENABLE_MURB_CUDA}'")
message(STATUS "MUrB info: ")
message(STATUS "  * CMAKE_BUILD_TYPE: '${CMAKE_BUILD_TYPE}'")

# Set the path to the CMake modules
set (CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules/")

# Specify bin path
set (EXECUTABLE_OUTPUT_PATH bin/)

# Enable CUDA (nvcc)
if (ENABLE_MURB_CUDA)
    include(CheckLanguage)
    check_language(CUDA)
    enable_language(CUDA)
    include_directories(${CUDA_INCLUDE_DIRS})
    if (NOT DEFINED CMAKE_CUDA_STANDARD)
        set(CMAKE_CUDA_STANDARD 11)
        set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    endif ()
else ()
    # Enable C++11
    set (CMAKE_CXX_STANDARD 11)
    set (CMAKE_CXX_STANDARD_REQUIRED ON)
endif ()

if (ENABLE_VISU)
    set (OpenGL_GL_PREFERENCE LEGACY)
    find_package (OpenGL)
    find_package (GLEW)
    find_package (GLM)
    find_package (GLFW)

    if ((OPENGL_FOUND) AND (GLEW_FOUND) AND (GLM_FOUND) AND (GLFW_FOUND))
        set (LINK_VISU 1)
    else ()
        message(STATUS "The visualization has been disabled.")
        set (LINK_VISU 0)
    endif ()
endif ()

# Generate the common source code files list
file (GLOB_RECURSE source_common src/common/*)

# Object
add_library (common-lib OBJECT ${source_common})
list(APPEND murb_targets_list common-lib)

if (ENABLE_MURB)
    # Generate the simulator source code files list
    file (GLOB_RECURSE source_murb_implem_files src/murb/implem/*)

    # MUrB objects
    add_library (murb-implem-lib OBJECT ${source_murb_implem_files})
    list(APPEND murb_targets_list murb-implem-lib)

    # Executable declaration
    file (GLOB_RECURSE source_murb_main_file src/murb/main.cpp)
    add_executable (murb-bin $<TARGET_OBJECTS:common-lib> $<TARGET_OBJECTS:murb-implem-lib> ${source_murb_main_file})
    set_target_properties (murb-bin PROPERTIES OUTPUT_NAME murb)
    list(APPEND murb_targets_list murb-bin)

    # set 32-bit floating point precision
    target_compile_definitions(murb-bin PUBLIC NBODY_FLOAT)

    if (ENABLE_TEST)
        file (GLOB_RECURSE source_test_files src/test/*)
        add_executable (test-bin $<TARGET_OBJECTS:common-lib> $<TARGET_OBJECTS:murb-implem-lib> ${source_test_files})
        set_target_properties (test-bin PROPERTIES OUTPUT_NAME murb-test)
        list(APPEND murb_targets_list test-bin)
        # include Catch2 header
        target_include_directories (test-bin PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/lib/Catch2/include/")
        # include MUrB header
        target_include_directories (test-bin PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src/murb/implem")
        enable_testing()
        add_test(NAME murb::test COMMAND test-bin)
    endif ()
endif ()

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

targets_include_directories ("${murb_targets_list}" PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src/common")

# MIPP wrapper includes
if (EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/lib/MIPP/src/mipp.h")
    targets_compile_definitions ("${murb_targets_list}" PRIVATE MIPP_ENABLE_BACKTRACE)
    targets_include_directories ("${murb_targets_list}" PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/lib/MIPP/src/")
    message(STATUS "MIPP found")
else ()
    message(FATAL_ERROR "MIPP can't be found, try to init the submodules with the following cmd:\n"
                        "$ git submodule update --init -- ../lib/MIPP/")
endif ()

if (LINK_VISU)
    message(STATUS "Link with OpenGL libraries")
    if (OPENGL_FOUND)
        targets_link_libraries ("${murb_targets_list}" PRIVATE OpenGL::GL)
#       targets_link_libraries ("${murb_targets_list}" PRIVATE OpenGL::GLX)
    endif (OPENGL_FOUND)

    if (GLEW_FOUND)
        targets_include_directories ("${murb_targets_list}" PRIVATE ${GLEW_INCLUDE_DIRS})
        targets_link_libraries ("${murb_targets_list}" PRIVATE GLEW::GLEW)
    endif (GLEW_FOUND)

    if (GLM_FOUND)
        targets_include_directories ("${murb_targets_list}" PRIVATE ${GLM_INCLUDE_DIRS})
    endif (GLM_FOUND)

    if (GLFW_FOUND)
        targets_include_directories ("${murb_targets_list}" PRIVATE ${GLFW_INCLUDE_DIR})
        targets_link_libraries ("${murb_targets_list}" PRIVATE "${GLFW_LIBRARIES}")
    endif (GLFW_FOUND)

    # Enable visualization
    targets_compile_definitions("${murb_targets_list}" PRIVATE VISU)
endif ()

if (ENABLE_MURB_OMP)
    find_package (OpenMP REQUIRED)
    if (OpenMP_FOUND)
        targets_link_libraries("${murb_targets_list}" PUBLIC OpenMP::OpenMP_CXX)
    endif (OpenMP_FOUND)
endif()

if ((ENABLE_MURB_OCL) AND (ENABLE_MURB))
    # Enable OCL
    targets_compile_definitions("${murb_targets_list}" PRIVATE USE_OCL)

    find_package (OpenCL REQUIRED)
    if (OpenCL_FOUND)
        message(STATUS "OpenCL found")
        targets_link_libraries("${murb_targets_list}" PUBLIC OpenCL::OpenCL)
    endif (OpenCL_FOUND)
endif ()

if ((ENABLE_MURB_CUDA) AND (ENABLE_MURB))
    targets_compile_definitions("${murb_targets_list}" PRIVATE USE_CUDA)
endif ()

