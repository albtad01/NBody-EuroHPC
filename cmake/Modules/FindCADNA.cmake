find_path(CADNA_INCLUDE_DIR
    NAMES cadna.h
    PATHS
        ENV CADNA_ROOT
        /usr/local
        /usr
    PATH_SUFFIXES include
)

find_library(CADNA_LIBRARY
    NAMES cadnaCdebug cadnaC
    PATHS
        ENV CADNA_ROOT
        /usr/local
        /usr
    PATH_SUFFIXES lib
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(CADNA
    REQUIRED_VARS CADNA_INCLUDE_DIR CADNA_LIBRARY
)

if (CADNA_FOUND)
    set(CADNA_INCLUDE_DIRS ${CADNA_INCLUDE_DIR})
    set(CADNA_LIBRARIES ${CADNA_LIBRARY})

    if (NOT TARGET CADNA::CADNA)
        add_library(CADNA::CADNA UNKNOWN IMPORTED)
        set_target_properties(CADNA::CADNA PROPERTIES
            IMPORTED_LOCATION "${CADNA_LIBRARY}"
            INTERFACE_INCLUDE_DIRECTORIES "${CADNA_INCLUDE_DIR}"
        )
    endif ()
endif ()
