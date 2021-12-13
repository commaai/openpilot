########################################################
#   -DSPHINXOPTS : build options to pass to `sphinx-build`
#       https://www.sphinx-doc.org/en/master/man/sphinx-build.html#options
########################################################

find_package(Sphinx REQUIRED)

option(SPHINXOPTS "sphinx-build options" "")
set(ENV{SPHINXOPTS} ${SPHINXOPTS})
set(SPHINX_BUILDDIR ${OPENPILOT_ROOT}/docs/build/sphinx/html CACHE INTERNAL "Sphinx build output path" )

# vars to sub in .in files
set(conf_path "${CMAKE_CURRENT_SOURCE_DIR}")

execute_process(
    COMMAND git rev-parse --short HEAD
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    OUTPUT_VARIABLE GIT_COMMIT_SHA
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

# sphinx config
set(CONF_IN ${CMAKE_CURRENT_SOURCE_DIR}/conf.py.in)
set(CONF_OUT ${CMAKE_CURRENT_BINARY_DIR}/conf.py)
configure_file( ${CONF_IN} ${CONF_OUT} @ONLY)

# homepage
set(INDEX_IN ${CMAKE_CURRENT_SOURCE_DIR}/index.md.in)
set(INDEX_OUT ${CMAKE_CURRENT_BINARY_DIR}/index.md)
configure_file(${INDEX_IN} ${INDEX_OUT} @ONLY)

# markdown sub-pages
set(RST_IN ${CMAKE_CURRENT_SOURCE_DIR}/overview.rst.in)
set(RST_OUT ${CMAKE_CURRENT_BINARY_DIR}/overview.rst)
configure_file( ${RST_IN} ${RST_OUT} @ONLY)

# c doxygen docs
set(CDOCS_RST_IN ${CMAKE_CURRENT_SOURCE_DIR}/c_docs.rst.in)
set(CDOCS_RST_OUT ${CMAKE_CURRENT_BINARY_DIR}/c_docs.rst)
configure_file( ${CDOCS_RST_IN} ${CDOCS_RST_OUT} @ONLY)

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
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/sphinx-config.sh
        ${OPENPILOT_ROOT}
        ${CMAKE_CURRENT_BINARY_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}
)

# build rst & html files (main target)
add_custom_command(
    OUTPUT rst_html
    COMMAND ${SPHINX_API_EXEC} -o ${CMAKE_CURRENT_BINARY_DIR} ${OPENPILOT_ROOT} ${SPHINX_API_RST_EXCLUDE}
    COMMAND ${SPHINX_BUILD_EXEC} -b html -T -v ${CMAKE_CURRENT_BINARY_DIR} ${CMAKE_BINARY_DIR}/html
        # -Dbreathe_projects.c_docs=${CMAKE_BINARY_DIR}/html/xml/breathe/doxygen    # document all doxygen-generated docs
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "building rst and html files..."
    VERBATIM
)

add_custom_target(sphinx_build
    ALL
    DEPENDS rst_html
)