#pragma once

#include <memory>

#include "common/util.h"
#include "system/camerad/cameras/camera_common.h"
#include "system/camerad/cameras/hw.h"
#include "system/camerad/sensors/sensor.h"

// V4L2 camera backend for mainline CAMSS kernel driver.
// Used instead of Spectra when running on a mainline kernel.

struct V4L2CameraConfig {
  const char *sensor_name;    // e.g. "ox03c10 16-0036"
  const char *csiphy_name;    // e.g. "msm_csiphy0"
  const char *csid_name;      // e.g. "msm_csid0"
  const char *vfe_pix_name;   // e.g. "msm_vfe0_pix"
  int video_dev_index;        // e.g. 3 for /dev/video3
};

class V4L2Camera {
public:
  V4L2Camera(const CameraConfig &config, const V4L2CameraConfig &v4l2_config);
  ~V4L2Camera();

  void camera_open(VisionIpcServer *v);
  void camera_close();
  void sensors_start();
  int capture_frame();  // poll + DQBUF, returns buf index or -1
  void queue_buffer(int idx);
  void set_exposure(int exposure_time, int analogue_gain);

  bool enabled;
  CameraConfig cc;
  std::unique_ptr<const SensorInfo> sensor;
  CameraBuf buf;

  uint32_t stride;
  uint32_t y_height;
  uint32_t uv_height;
  uint32_t uv_offset;
  uint32_t yuv_size;

  int video_fd = -1;  // public for poll()

private:
  V4L2CameraConfig v4l2_cc;
  int media_fd = -1;
  int sensor_subdev_fd = -1;

  static constexpr int NUM_BUFFERS = VIPC_BUFFER_COUNT;
  void *mmap_bufs[NUM_BUFFERS] = {};
  size_t mmap_lengths[NUM_BUFFERS] = {};

  void setup_media_links();
  void setup_formats();
  int find_video_device();
  int find_sensor_subdev();
};

// V4L2 camera main thread, called from camerad_thread() on mainline kernel
void camerad_thread_v4l2();

// Returns true if running on mainline kernel with CAMSS V4L2 driver
bool is_mainline_camss();
