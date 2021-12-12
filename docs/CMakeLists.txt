########################################################################
# CMake build nstructions for https://docs.comma.ai/
#
# instructions (local build): opening `build/html/index.html` in a browser
#   mkdir build && cd build && cmake ../ && cmake --build . -- -j$(nproc)
#
# instructions (docker build): http://localhost:3000/
#   docker build -t ghcr.io/commaai/openpilot-docs:latest -f docs/docker/Dockerfile ..
#   docker run --rm -d 3000:80 ghcr.io/commaai/openpilot-docs:latest
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
set(OPENPILOT_ROOT ${OPENPILOT_ROOT} CACHE INTERNAL "openpilot root directory")

# add_subdirectory(doxygen)
add_subdirectory(sphinx)