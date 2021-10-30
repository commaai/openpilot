if(NOT SPHINX_DEPS_FOUND)

    # set(SEARCH_CMD "find . -type f \( -name '*.md' -o -name '*.rst' -o -name '*.png' -o -name '*.jpg' \) \
    #     -not -path '*/.*' \
    #     -not -path './build/*' \
    #     -not -path './docs/*' \
    #     -not -path './xx/*'"
    # )

    message(STATUS "Searching for Sphinx docs & configs...")
    file(GLOB_RECURSE SPHINX_DEPS_SRCS
        "${OPENPILOT_ROOT}/*.md"
        "${OPENPILOT_ROOT}/*.rst"
        "${OPENPILOT_ROOT}/*.png"
        "${OPENPILOT_ROOT}/*.jpg"
    )
    list(FILTER ${SPHINX_DEPS_SRCS} EXCLUDE REGEX "*/*")
    list(FILTER ${SPHINX_DEPS_SRCS} EXCLUDE REGEX "\./build/*")
    list(FILTER ${SPHINX_DEPS_SRCS} EXCLUDE REGEX "\./docs/*")
    list(FILTER ${SPHINX_DEPS_SRCS} EXCLUDE REGEX "\./xx/*")

    # execute_process(
    #     COMMAND bash -c ${SEARCH_CMD}
    #     RESULTS_VARIABLE SPHINX_DEPENDS
    # )

    if(SPHINX_DEPS_SRCS)

        set(SPHINX_DEPS_FOUND TRUE CACHE INTERNAL "found sphinx sources")
        message(STATUS "Found sphinx_deps: ${SPHINX_SRCS}")
    endif(SPHINX_DEPS_SRCS)

    mark_as_advanced(SPHINX_SRCS)
endif(NOT SPHINX_DEPS_FOUND)