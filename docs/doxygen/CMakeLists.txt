find_package(Doxygen REQUIRED)

file(TO_NATIVE_PATH ${OPENPILOT_ROOT} top_srcdir)
file(TO_NATIVE_PATH ${CMAKE_BINARY_DIR} top_builddir)
file(TO_NATIVE_PATH ${OPENPILOT_ROOT} abs_top_srcdir)
file(TO_NATIVE_PATH ${CMAKE_BINARY_DIR} abs_top_builddir)

set(HAVE_DOT ${DOXYGEN_DOT_FOUND} CACHE INTERNAL "graphviz is installed to generate doxy graphics")

set(DOXYGEN_IN ${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.in)
set(DOXYGEN_OUT ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile)

configure_file(
    ${DOXYGEN_IN}
    ${DOXYGEN_OUT}
    @ONLY
)

set(DOXY_BUILD_DIR ${CMAKE_CURRENT_BINARY_DIR}/xml CACHE INTERNAL "Path to doxygen build files")

########################################################################
# setup doxy_build target
# generates c/c++ .xml files -> copies to root html/xml
########################################################################
add_custom_command(
    OUTPUT doxy_xml
    COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYGEN_OUT}
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_BINARY_DIR}/xml ${CMAKE_BINARY_DIR}/html/xml
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "generating C/C++ documentation with doxygen..."
    VERBATIM
)

add_custom_target(doxy_build
    ALL
    DEPENDS doxy_xml
)