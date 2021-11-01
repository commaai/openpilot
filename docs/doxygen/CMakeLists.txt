set(LOGO_IMAGE "${CMAKE_CURRENT_SOURCE_DIR}/img/logo.png" CACHE INTERNAL "doxygen logo path" FORCE)
set(IMAGE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/img/" CACHE INTERNAL "doxygen misc images path" FORCE)

file(TO_NATIVE_PATH ${OPENPILOT_ROOT} top_srcdir)
file(TO_NATIVE_PATH ${CMAKE_BINARY_DIR} top_builddir)
file(TO_NATIVE_PATH ${OPENPILOT_ROOT} abs_top_srcdir)
file(TO_NATIVE_PATH ${CMAKE_BINARY_DIR} abs_top_builddir)

set(HAVE_DOT ${DOXYGEN_DOT_FOUND} CACHE INTERNAL "graphviz is installed to generate doxy graphics")

set(DOXYGEN_IN ${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile)
set(DOXYGEN_OUT ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile.out)

configure_file(
    ${DOXYGEN_IN}
    ${DOXYGEN_OUT}
    @ONLY
)

set(DOXY_BUILD_DIR ${CMAKE_CURRENT_BINARY_DIR}/xml CACHE INTERNAL "Path to doxygen build files")

########################################################################
# setup doxy_build target
########################################################################
add_custom_command(
    OUTPUT ${DOXY_BUILD_DIR}
    COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYGEN_OUT}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "generating C/C++ documentation with doxygen..."
    VERBATIM
)

add_custom_target(doxy_build
    ALL
    DEPENDS ${DOXY_BUILD_DIR}
)

install(DIRECTORY ${DOXY_BUILD_DIR} DESTINATION ${OPENPILOT_ROOT}/build/docs)