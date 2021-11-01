########################################################################
# setup sphinx_build
########################################################################
set(SPHINX_CONF_IN ${CMAKE_CURRENT_SOURCE_DIR}/conf.py)
set(SPHINX_CONF_OUT ${CMAKE_CURRENT_BINARY_DIR}/conf.py)

configure_file(
    ${SPHINX_CONF_IN}
    ${SPHINX_CONF_OUT}
    @ONLY
)

set(SPHINX_OVERVIEW_IN ${CMAKE_CURRENT_SOURCE_DIR}/overview.rst)
set(SPHINX_OVERVIEW_OUT ${CMAKE_CURRENT_BINARY_DIR}/overview.rst)

configure_file(
    ${SPHINX_OVERVIEW_IN}
    ${SPHINX_OVERVIEW_OUT}
    @ONLY
)

## FIXME doesn't copy using parent directory, so files overwrite eachother >_>
# copy SPHINX_DEPS_SRCS files to sphinx build dir before building
execute_process(
    COMMAND ${CMAKE_COMMAND} -E copy ${SPHINX_DEPS_SRCS} ${CMAKE_CURRENT_BINARY_DIR}
)

add_custom_command(
    OUTPUT ${SPHINX_BUILDDIR}
    COMMAND ${SPHINX_RST_EXEC} -o ${SPHINX_BUILDDIR} ${OPENPILOT_ROOT} ${SPHINX_RST_EXCLUDE_SRCS}
    COMMAND ${SPHINX_EXEC} -T -v ${CMAKE_CURRENT_BINARY_DIR} ${SPHINX_BUILDDIR} ${SPHINXOPTS}
    -Dbreathe_projects.c_docs=${DOXY_BUILD_DIR}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "generating rst files for sphinx..."
    VERBATIM
)

add_custom_target(sphinx_build
    ALL
    DEPENDS ${SPHINX_BUILDDIR}
)

install(DIRECTORY ${SPHINX_BUILDDIR} DESTINATION ${OPENPILOT_ROOT}/build/docs)