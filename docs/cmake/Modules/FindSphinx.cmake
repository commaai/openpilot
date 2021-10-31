########################################################################
# Sets the following cmake variables when find_package(Sphinx) is called:
#  - SPHINX_FOUND : TRUE if SPHINX binaries are found, FALSE otherwise
#  - SPHINX_EXEC : path to `sphinx-build`
#  - SPHINX_RST_EXEC : path to `sphinx-apidoc`
#  - SPHINXOPTS : options to pass to `sphinx-build`
#  - SPHINX_DOCSDIR : path to the Sphinx source documentation directory
#  - SPHINX_BUILDDIR : path to the Sphinx build output directory
#  - SPHINX_DEPS_SRCS : list of sources to be used to be built by Sphinx
#  - SPHINX_RST_SRCS : list of sources to be used to build .rst files, by Sphinx
########################################################################

if(NOT SPHINX_FOUND)

    find_program(SPHINX_EXEC
        NAMES sphinx-build
        DOC "path to sphinx-build executable"
    )

    # Handle standard arguments to find_package like REQUIRED and QUIET
    include(FindPackageHandleStandardArgs)
    find_package_handle_standard_args(Sphinx "Failed to find sphinx-build executable" SPHINX_EXEC )
    find_package_handle_standard_args(Sphinx "Failed to find sphinx-build executable" SPHINX_RST_EXEC )

    find_program(SPHINX_RST_EXEC
        NAMES sphinx-apidoc
        DOC "path to sphinx-apidoc executable, to generate .rst files"
    )

    if(SPHINX_EXEC AND SPHINX_RST_EXEC)
        set(SPHINX_FOUND TRUE CACHE INTERNAL "Found Sphinx execs")
    endif(SPHINX_EXEC AND SPHINX_RST_EXEC)

endif(NOT SPHINX_FOUND)
mark_as_advanced(SPHINX_FOUND)

option(SPHINXOPTS "Sphinx build flag options" "")
option(SPHINX_DOCSDIR "Sphinx source docs path" ${OPENPILOT_ROOT}/docs/sphinx)
option(SPHINX_BUILDDIR "Sphinx build output path" ${OPENPILOT_ROOT}/docs/build/sphinx)

# message(STATUS "Searching for Sphinx .rst docs & configs...")
file(GLOB_RECURSE SPHINX_DEPS_SRCS
    "${OPENPILOT_ROOT}/*.md"
    "${OPENPILOT_ROOT}/*.rst"
    "${OPENPILOT_ROOT}/*.png"
    "${OPENPILOT_ROOT}/*.jpg"
)

if(SPHINX_DEPS_SRCS)
    # message(STATUS "Found sphinx_deps: ${SPHINX_DEPS_SRCS}")      # debug
    set(SPHINX_DEPS_SRCS ${SPHINX_DEPS_SRCS} CACHE INTERNAL ".rst build sources")
endif(SPHINX_DEPS_SRCS)

list(APPEND SPHINX_RST_SRCS
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
    OUTPUT_VARIABLE SPHINX_RST_SRCS_XTRA
    # COMMAND_ECHO STDOUT
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

# convert bash `find` output to usable cmake 'list' variable
string(REPLACE "\n" ";" SPHINX_RST_SRCS_XTRA "${SPHINX_RST_SRCS_XTRA}")
# message(STATUS "SPHINX_RST_SRCS_XTRA: ${SPHINX_RST_SRCS_XTRA}") # DEBUG

list(APPEND SPHINX_RST_SRCS ${SPHINX_RST_SRCS_XTRA})
set(SPHINX_RST_SRCS ${SPHINX_RST_SRCS} CACHE INTERNAL "sphinx .rst build sources")
# message(STATUS "SPHINX_RST_SRCS: ${SPHINX_RST_SRCS}")   # DEBUG