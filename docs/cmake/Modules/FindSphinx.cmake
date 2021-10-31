########################################################################
# Sets the following cmake variables when find_package(Sphinx) is called:
#  - SPHINX_FOUND : TRUE if SPHINX binaries are found, FALSE otherwise
#  - SPHINX_EXEC : path to `sphinx-build`
#  - SPHINX_RST_EXEC : path to `sphinx-apidoc`
#  - SPHINXOPTS : options to pass to `sphinx-build`
#  - SPHINX_DOCSDIR : path to the Sphinx source documentation directory
#  - SPHINX_BUILDDIR : path to the Sphinx build output directory
#  - SPHINX_DEPS_SRCS : list of sources to be used to be built by Sphinx
#  - SPHINX_RST_EXCLUDE_SRCS : list of dirs to be ignored when building .rst modules, by Sphinx
########################################################################

# Handle standard arguments to find_package like REQUIRED and QUIET
include(FindPackageHandleStandardArgs)

if(NOT SPHINX_FOUND)

    find_program(SPHINX_EXEC
        NAMES sphinx-build
        DOC "path to sphinx-build executable"
    )

    find_package_handle_standard_args(Sphinx "Failed to find sphinx-build executable" SPHINX_EXEC )

    find_program(SPHINX_RST_EXEC
        NAMES sphinx-apidoc
        DOC "path to sphinx-apidoc executable, to generate .rst files"
    )

    find_package_handle_standard_args(Sphinx "Failed to find sphinx-apidoc executable" SPHINX_RST_EXEC )

    if(SPHINX_EXEC AND SPHINX_RST_EXEC)
        set(SPHINX_FOUND TRUE CACHE INTERNAL "Found Sphinx execs")
    endif(SPHINX_EXEC AND SPHINX_RST_EXEC)

    mark_as_advanced(SPHINX_FOUND)
endif(NOT SPHINX_FOUND)

set(SPHINXOPTS "" CACHE INTERNAL "Sphinx build flag options")
set(SPHINX_DOCSDIR ${OPENPILOT_ROOT}/docs/sphinx CACHE INTERNAL "Sphinx source docs path" )
set(SPHINX_BUILDDIR ${OPENPILOT_ROOT}/docs/build/sphinx/html CACHE INTERNAL "Sphinx build output path" )

# message(STATUS "Searching for Sphinx .rst docs & configs...")
set(sphinx_deps_search_cmd "find ${OPENPILOT_ROOT} -type f \\( -name '*.md' -o -name '*.rst' -o -name '*.png' -o -name '*.jpg' \\) \\
    -not -path '*/.*'   \\
    -not -path './build/*' \\
    -not -path './docs/*' \\
    -not -path './xx/*'"
)

execute_process(
    COMMAND bash -c ${sphinx_deps_search_cmd}
    OUTPUT_VARIABLE SPHINX_DEPS_SRCS
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

string(REPLACE "\n" ";" SPHINX_DEPS_SRCS "${SPHINX_DEPS_SRCS}")
set(SPHINX_DEPS_SRCS ${SPHINX_DEPS_SRCS} CACHE INTERNAL "sphinx base config files")
# message(STATUS "DEPS_SRCS ${SPHINX_DEPS_SRCS}") # debug


list(APPEND SPHINX_RST_EXCLUDE_SRCS
    ${OPENPILOT_ROOT}/xx
    ${OPENPILOT_ROOT}/laika_repo
    ${OPENPILOT_ROOT}/rednose_repo
    ${OPENPILOT_ROOT}/pyextra
    ${OPENPILOT_ROOT}/notebooks
    ${OPENPILOT_ROOT}/panda_jungle
    ${OPENPILOT_ROOT}/third_party
    ${OPENPILOT_ROOT}/panda/examples
    ${OPENPILOT_ROOT}/scripts
    ${OPENPILOT_ROOT}/selfdrive/modeld
    ${OPENPILOT_ROOT}/selfdrive/debug
)

set(rst_xtra_search_cmd "find ${OPENPILOT_ROOT} -type d -name '*test*'")
execute_process(
    COMMAND bash -c ${rst_xtra_search_cmd}
    OUTPUT_VARIABLE SPHINX_RST_EXCLUDE_SRCS_XTRA
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

string(REPLACE "\n" ";" SPHINX_RST_EXCLUDE_SRCS_XTRA "${SPHINX_RST_EXCLUDE_SRCS_XTRA}")
# message(STATUS "SPHINX_RST_EXCLUDE_SRCS_XTRA: ${SPHINX_RST_EXCLUDE_SRCS_XTRA}") # DEBUG

list(APPEND SPHINX_RST_EXCLUDE_SRCS ${SPHINX_RST_EXCLUDE_SRCS_XTRA})
set(SPHINX_RST_EXCLUDE_SRCS ${SPHINX_RST_EXCLUDE_SRCS} CACHE INTERNAL "sphinx .rst build sources")
# message(STATUS "SPHINX_RST_EXCLUDE_SRCS: ${SPHINX_RST_EXCLUDE_SRCS}")   # DEBUG