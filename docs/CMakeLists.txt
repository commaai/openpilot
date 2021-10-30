cmake_minimum_required(VERSION 3.10)
project(docs
    DESCRIPTION "comma.ai developer documentation"
    HOMEPAGE_URL https://docs.comma.ai/
)

find_package(PkgConfig REQUIRED)
list(INSERT CMAKE_MODULE_PATH 0 ${CMAKE_SOURCE_DIR}/cmake/Modules)

# set OPENPILOT_ROOT
find_package(Git QUIET)
execute_process(
    COMMAND git rev-parse --show-toplevel
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    OUTPUT_VARIABLE OPENPILOT_ROOT
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
set(OPENPILOT_ROOT ${OPENPILOT_ROOT} CACHE STRING "openpilot absolute root directory")

find_package(Doxygen REQUIRED)

# set these variables using traditional `-DVARIABLE=VALUE`
option(SPHINXOPTS "Sphinx options" "")
option(SPHINXBUILD "" "sphinx-build")
option(DOCSDIR "" ${OPENPILOT_ROOT}/docs)
option(SOURCEDIR "" ${OPENPILOT_ROOT}/build/docs)
option(DOCSBUILDDIR "" ${OPENPILOT_ROOT}/build/docs)
option(BUILDDIR "" ${OPENPILOT_ROOT}/build)


find_package(SphinxDepends REQUIRED)

list(APPEND rst_srcs
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

set(RST_XTRA_SEARCH_CMD "find .. -type d -name '*test*'")

execute_process(
    COMMAND bash -c ${RST_XTRA_SEARCH_CMD}
    RESULTS_VARIABLE rst_srcs_xtra
)

list(APPEND rst_srcs ${rst_srcs_xtra})

# build rst files...
message(STATUS "Building rst files...")
add_custom_command(
    OUTPUT ${DOCSBUILDDIR}
    COMMAND ${SPHINXBUILD} -M "docs" ${SOURCEDIR} ${DOCSBUILDDIR} ${SPHINXOPTS}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Generating C/C++ documentation with doxygen"
    VERBATIM
)

add_custom_target(sphinx_docs
    ALL
    DEPENDS ${DOCSBUILDDIR}
)


# build html files...
add_subdirectory(doxygen)
