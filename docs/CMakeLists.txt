########################################################################
# CMake instructions to build https://docs.comma.ai/
#
# instructions (local build): opening `index.html`
#   mkdir build && cd build && cmake ../ && cmake --build . -- -j$(nproc)
#
# instructions (docker build): http://localhost:3000/
#   docker build -t ghcr.io/commaai/openpilot-docs:latest -f docker/Dockerfile ..
#   docker run --rm -d 3000:80 ghcr.io/commaai/openpilot-docs:latest
#
#   on non-amd64 systems (needs github registry access via personal access token):
#       docker buildx build --platform linux/amd64 -t ghcr.io/<github_username>/openpilot-docs:latest -f docker/Dockerfile .. -- push
#       ddocker run --rm -d 3000:80 ghcr.io/<github_username>/openpilot-docs:latest
########################################################################

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
set(OPENPILOT_ROOT ${OPENPILOT_ROOT} CACHE STRING "openpilot root directory")
set(CMAKE_INSTALL_PREFIX ${OPENPILOT_ROOT})

find_package(Sphinx REQUIRED)
find_package(Doxygen REQUIRED)

add_subdirectory(doxygen)
add_subdirectory(sphinx)