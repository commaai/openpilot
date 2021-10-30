if(NOT SPHINX_DEPS_FOUND)

    message(STATUS "Searching for Sphinx docs & configs...")
    file(GLOB_RECURSE SPHINX_DEPS_SRCS
        "${OPENPILOT_ROOT}/*.md"
        "${OPENPILOT_ROOT}/*.rst"
        "${OPENPILOT_ROOT}/*.png"
        "${OPENPILOT_ROOT}/*.jpg"
    )

    if(SPHINX_DEPS_SRCS)
        set(SPHINX_DEPS_FOUND TRUE CACHE INTERNAL "found sphinx sources")
        # message(STATUS "Found sphinx_deps: ${SPHINX_DEPS_SRCS}")      # debug
        set(SPHINX_DEPS_SRCS ${SPHINX_DEPS_SRCS} CACHE INTERNAL "Sphinx Sources")
    endif(SPHINX_DEPS_SRCS)

    mark_as_advanced(SPHINX_SRCS)
endif(NOT SPHINX_DEPS_FOUND)