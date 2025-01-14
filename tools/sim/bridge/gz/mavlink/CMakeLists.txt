cmake_minimum_required(VERSION 3.15)
project(openpilot-mavlink-bridge)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Enable strict handling of warnings
add_definitions("-Wall -Wextra -Werror")

# Finds MAVSDK when installed system wide.
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
    #  autopilot_interface.cpp
    #  serial_port.cpp
    #  udp_port.cpp
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
    #$ENV{OPENPILOT_DIR}/mavlink_repo
)
