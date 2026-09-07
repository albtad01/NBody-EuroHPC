# Runs at the beginning of every build. A failed build must not leave a ready stamp.
file(MAKE_DIRECTORY "${OUTPUT_DIR}")
file(REMOVE "${OUTPUT_DIR}/../murb-build.ready")
execute_process(COMMAND git -C "${SOURCE_DIR}" rev-parse HEAD
    OUTPUT_VARIABLE revision OUTPUT_STRIP_TRAILING_WHITESPACE RESULT_VARIABLE git_result)
if(NOT git_result EQUAL 0)
    set(revision "unknown")
endif()
execute_process(COMMAND git -C "${SOURCE_DIR}" status --porcelain --untracked-files=normal
    OUTPUT_VARIABLE changes OUTPUT_STRIP_TRAILING_WHITESPACE RESULT_VARIABLE status_result)
set(dirty 0)
if(NOT status_result EQUAL 0 OR NOT changes STREQUAL "")
    set(dirty 1)
endif()
file(CONFIGURE OUTPUT "${OUTPUT_DIR}/BuildInfo.hpp" CONTENT
    "#pragma once\n#define MURB_REVISION \"${revision}\"\n#define MURB_BUILD_DIRTY ${dirty}\n" @ONLY)
