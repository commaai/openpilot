#include "system/camerad/cameras/utils.h"

#include <sys/ioctl.h>

#include "common/swaglog.h"
#include "common/util.h"
#include "media/cam_defs.h"
#include "media/cam_sync.h"

int do_cam_control(int fd, int op_code, void *handle, int size) {
  struct cam_control camcontrol = {0};
  camcontrol.op_code = op_code;
  camcontrol.handle = (uint64_t)handle;
  if (size == 0) {
    camcontrol.size = 8;
    camcontrol.handle_type = CAM_HANDLE_MEM_HANDLE;
  } else {
    camcontrol.size = size;
    camcontrol.handle_type = CAM_HANDLE_USER_POINTER;
  }

  int ret = HANDLE_EINTR(ioctl(fd, VIDIOC_CAM_CONTROL, &camcontrol));
  if (ret == -1) {
    LOGE("VIDIOC_CAM_CONTROL error: op_code %d - errno %d", op_code, errno);
  }
  return ret;
}

int do_sync_control(int fd, uint32_t id, void *handle, uint32_t size) {
  struct cam_private_ioctl_arg arg = {
      .id = id,
      .size = size,
      .ioctl_ptr = (uint64_t)handle,
  };
  int ret = HANDLE_EINTR(ioctl(fd, CAM_PRIVATE_IOCTL_CMD, &arg));

  int32_t ioctl_result = static_cast<int32_t>(arg.result);
  if (ret < 0) {
    LOGE("CAM_SYNC error: id %u - errno %d - ret %d - ioctl_result %d", id, errno, ret, ioctl_result);
    return ret;
  }
  if (ioctl_result != 0) {
    LOGE("CAM_SYNC error: id %u - errno %d - ret %d - ioctl_result %d", id, errno, ret, ioctl_result);
    return ioctl_result;
  }
  return ret;
}

std::optional<int32_t> device_acquire(int fd, int32_t session_handle, void *data, uint32_t num_resources) {
  struct cam_acquire_dev_cmd cmd = {
      .session_handle = session_handle,
      .handle_type = CAM_HANDLE_USER_POINTER,
      .num_resources = (uint32_t)(data ? num_resources : 0),
      .resource_hdl = (uint64_t)data,
  };
  int err = do_cam_control(fd, CAM_ACQUIRE_DEV, &cmd, sizeof(cmd));
  return err == 0 ? std::make_optional(cmd.dev_handle) : std::nullopt;
}

int device_config(int fd, int32_t session_handle, int32_t dev_handle, uint64_t packet_handle) {
  struct cam_config_dev_cmd cmd = {
      .session_handle = session_handle,
      .dev_handle = dev_handle,
      .packet_handle = packet_handle,
  };
  return do_cam_control(fd, CAM_CONFIG_DEV, &cmd, sizeof(cmd));
}

int device_control(int fd, int op_code, int session_handle, int dev_handle) {
  // start stop and release are all the same
  struct cam_start_stop_dev_cmd cmd{.session_handle = session_handle, .dev_handle = dev_handle};
  return do_cam_control(fd, op_code, &cmd, sizeof(cmd));
}

int open_v4l_by_name_and_index(const char name[], int index, int flags) {
  for (int v4l_index = 0; /**/; ++v4l_index) {
    std::string v4l_name = util::read_file(util::string_format("/sys/class/video4linux/v4l-subdev%d/name", v4l_index));
    if (v4l_name.empty()) return -1;
    if (v4l_name.find(name) == 0) {
      if (index == 0) {
        return HANDLE_EINTR(open(util::string_format("/dev/v4l-subdev%d", v4l_index).c_str(), flags));
      }
      index--;
    }
  }
}
