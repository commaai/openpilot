#pragma once

#include <fcntl.h>

#include <cstdint>
#include <optional>

int open_v4l_by_name_and_index(const char name[], int index = 0, int flags = O_RDWR | O_NONBLOCK);
std::optional<int32_t> device_acquire(int fd, int32_t session_handle, void *data, uint32_t num_resources=1);
int device_config(int fd, int32_t session_handle, int32_t dev_handle, uint64_t packet_handle);
int device_control(int fd, int op_code, int session_handle, int dev_handle);
int do_cam_control(int fd, int op_code, void *handle, int size);
int do_sync_control(int fd, uint32_t id, void *handle, uint32_t size);

