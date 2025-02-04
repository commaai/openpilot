cmake_minimum_required(VERSION 3.15)
project(openpilot-mavlink-bridge)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Enable strict handling of warnings
add_definitions("-Wall -Wextra -Werror")

# Finds MAVSDK when installed system wide.
find_package(OpenCL REQUIRED)
find_package(gz-msgs10 REQUIRED)
find_package(gz-transport13 REQUIRED)
find_package(MAVSDK REQUIRED)

# Specify your app's executable name, and list of source files used to create it.
add_executable(mavlink-bridge
    mavlink-bridge.cpp
    $ENV{OPENPILOT_DIR}/cereal/gen/cpp/log.capnp.c++
    $ENV{OPENPILOT_DIR}/cereal/gen/cpp/car.capnp.c++
    $ENV{OPENPILOT_DIR}/cereal/gen/cpp/legacy.capnp.c++
    $ENV{OPENPILOT_DIR}/cereal/gen/cpp/custom.capnp.c++
)
# Specify your app's executable name and a list of linked libraries
target_link_libraries(mavlink-bridge
    MAVSDK::mavsdk
    gz-msgs10
    gz-transport13
    capnp
    kj
    pthread
)
target_include_directories(mavlink-bridge
    PUBLIC
    ${gz-msgs10_INCLUDE_DIRS}
    $ENV{OPENPILOT_DIR}
)

add_executable(gzimgpub
    gz-image-publisher.cpp
    $ENV{OPENPILOT_DIR}/cereal/gen/cpp/log.capnp.c++
    $ENV{OPENPILOT_DIR}/cereal/gen/cpp/car.capnp.c++
    $ENV{OPENPILOT_DIR}/cereal/gen/cpp/legacy.capnp.c++
    $ENV{OPENPILOT_DIR}/cereal/gen/cpp/custom.capnp.c++
)
target_compile_definitions(gzimgpub PRIVATE CL_TARGET_OPENCL_VERSION=300)
target_link_libraries(gzimgpub
    OpenCL::OpenCL
    gz-msgs10
    gz-transport13
    capnp
    kj
    pthread
)
target_include_directories(gzimgpub
    PUBLIC
    ${gz-msgs10_INCLUDE_DIRS}
    $ENV{OPENPILOT_DIR}
)
set_target_properties(gzimgpub PROPERTIES C_STANDARD 11
                                                 C_STANDARD_REQUIRED ON
                                                 C_EXTENSIONS OFF)
