set(LOGO_IMAGE "${CMAKE_CURRENT_SOURCE_DIR}/img/logo.png" CACHE INTERNAL "Doxygen Logo Path" FORCE)
set(IMAGE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/img/" CACHE INTERNAL "Doxygen Images Path" FORCE)

########################################################################
# Create the doxygen configuration file
########################################################################
file(TO_NATIVE_PATH ${OPENPILOT_ROOT} top_srcdir)
file(TO_NATIVE_PATH ${CMAKE_BINARY_DIR} top_builddir)
file(TO_NATIVE_PATH ${OPENPILOT_ROOT} abs_top_srcdir)
file(TO_NATIVE_PATH ${CMAKE_BINARY_DIR} abs_top_builddir)

set(HAVE_DOT ${DOXYGEN_DOT_FOUND})

set(DOXYGEN_IN ${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile)
set(DOXYGEN_OUT ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile.out)

configure_file(
    ${DOXYGEN_IN}
    ${DOXYGEN_OUT}
    @ONLY
)

set(DOXY_BUILD_DIR ${CMAKE_CURRENT_BINARY_DIR}/html)

add_custom_command(
    OUTPUT ${DOXY_BUILD_DIR}
    COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYGEN_OUT}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Generating C/C++ documentation with doxygen..."
    VERBATIM
)

add_custom_target(c_docs
    ALL
    DEPENDS ${DOXY_BUILD_DIR}
)