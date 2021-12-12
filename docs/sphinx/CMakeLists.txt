########################################################
#   -DSPHINXOPTS : build options to pass to `sphinx-build`
#       https://www.sphinx-doc.org/en/master/man/sphinx-build.html#options
########################################################

find_package(Sphinx REQUIRED)

option(SPHINXOPTS "sphinx-build options" "")
set(SPHINX_BUILDDIR ${OPENPILOT_ROOT}/docs/build/sphinx/html CACHE INTERNAL "Sphinx build output path" )

set(CONF_IN ${CMAKE_CURRENT_SOURCE_DIR}/conf.py)
set(CONF_OUT ${CMAKE_CURRENT_BINARY_DIR}/conf.py)

configure_file(
    ${CONF_IN}
    ${CONF_OUT}
    @ONLY
)

set(RST_IN ${CMAKE_CURRENT_SOURCE_DIR}/overview.rst)
set(RST_OUT ${CMAKE_CURRENT_BINARY_DIR}/overview.rst)

configure_file(
    ${RST_IN}
    ${RST_OUT}
    @ONLY
)

set(INDEX_IN ${CMAKE_CURRENT_SOURCE_DIR}/index.md)
set(INDEX_OUT ${CMAKE_CURRENT_BINARY_DIR}/index.md)

configure_file(
    ${INDEX_IN}
    ${INDEX_OUT}
    @ONLY
)

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


list(APPEND SPHINX_API_RST_EXCLUDE
    ${OPENPILOT_ROOT}/xx
    ${OPENPILOT_ROOT}/yy
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
    OUTPUT_VARIABLE SPHINX_API_RST_EXCLUDE_XTRA
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

string(REPLACE "\n" ";" SPHINX_API_RST_EXCLUDE_XTRA "${SPHINX_API_RST_EXCLUDE_XTRA}")
list(APPEND SPHINX_API_RST_EXCLUDE ${SPHINX_API_RST_EXCLUDE_XTRA})
set(SPHINX_API_RST_EXCLUDE ${SPHINX_API_RST_EXCLUDE} CACHE INTERNAL "sphinx .rst build sources")

# copy OP docs to build directory (keep the same file struct)
execute_process(
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/sphinx-config.sh ${OPENPILOT_ROOT} ${CMAKE_CURRENT_BINARY_DIR} ${CMAKE_CURRENT_SOURCE_DIR}
)

# build rst & html files (main target)
add_custom_command(
    OUTPUT rst_html
    COMMAND ${SPHINX_API_EXEC} -o ${CMAKE_CURRENT_BINARY_DIR} ${OPENPILOT_ROOT} ${SPHINX_API_RST_EXCLUDE}
    COMMAND ${SPHINX_BUILD_EXEC} -b html -T -v ${CMAKE_CURRENT_BINARY_DIR} ${CMAKE_CURRENT_BINARY_DIR}/html
        -Dbreathe_projects.c_docs=${DOXY_BUILD_DIR}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "generating rst files for sphinx..."
    VERBATIM
)

add_custom_target(sphinx_build
    ALL
    DEPENDS rst_html
)

install(DIRECTORY ${SPHINX_BUILDDIR} DESTINATION ${OPENPILOT_ROOT}/build/docs)