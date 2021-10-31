find_package(Sphinx REQUIRED)

########################################################################
# setup sphinx_rst target
########################################################################
add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}
    COMMAND ${SPHINX_RST_EXEC} -o ${CMAKE_CURRENT_BINARY_DIR} ../ ${SPHINX_RST_SRCS}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "generating rst files for sphinx..."
    VERBATIM
)

add_custom_target(sphinx_rst
    ALL
    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}
)

########################################################################
# setup sphinx_build target
# TODO: replace `SPHINX_DOCSDIR` and `SPHINX_BUILDDIR` with regular cmake paths
########################################################################
add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}
    COMMAND ${SPHINX_EXEC} -M ${PROJECT_NAME} ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_BINARY_DIR} ${SPHINXOPTS}
    -Dbreathe_projects.docs=${DOXY_BUILD_DIR}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "generating base sphinx html files..."
    VERBATIM
)

add_custom_target(sphinx_build
    ALL
    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}
)