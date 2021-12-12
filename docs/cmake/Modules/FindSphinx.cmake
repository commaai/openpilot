########################################################################
# Sets the following cmake variables when find_package(Sphinx) is called:
#  - SPHINX_FOUND : TRUE if SPHINX binaries are found, FALSE otherwise
#  - SPHINX_BUILD_EXEC : path to `sphinx-build`
#  - SPHINX_API_EXEC : path to `sphinx-apidoc`
########################################################################

# Handle standard arguments to find_package like REQUIRED and QUIET
include(FindPackageHandleStandardArgs)

if(NOT SPHINX_FOUND)

    find_program(SPHINX_BUILD_EXEC
        NAMES sphinx-build
        DOC "path to sphinx-build executable"
    )
    find_package_handle_standard_args(Sphinx "Failed to find sphinx-build executable" SPHINX_BUILD_EXEC )

    find_program(SPHINX_API_EXEC
        NAMES sphinx-apidoc
        DOC "path to sphinx-apidoc executable"
    )
    find_package_handle_standard_args(Sphinx "Failed to find sphinx-apidoc executable" SPHINX_API_EXEC )

    if(SPHINX_BUILD_EXEC AND SPHINX_API_EXEC)
        set(SPHINX_FOUND TRUE CACHE INTERNAL "Found Sphinx execs")
    endif(SPHINX_BUILD_EXEC AND SPHINX_API_EXEC)

    mark_as_advanced(SPHINX_FOUND)
endif(NOT SPHINX_FOUND)