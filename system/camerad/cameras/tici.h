#pragma once

#include "common/util.h"
#include "cereal/gen/cpp/log.capnp.h"
#include "msgq/visionipc/visionipc_server.h"

#include "media/cam_isp_ife.h"

// For the comma 3/3X three camera platform

struct CameraConfig {
  int camera_num;
  VisionStreamType stream_type;
  float focal_length_mm;  // millimeters
  const char *publish_name;
  cereal::FrameData::Builder (cereal::Event::Builder::*init_camera_state)();
  bool enabled;
  uint32_t phy;
  Rect expo_rect;
  float expo_focal_length;
};

// NOTE: to be able to disable road and wide road, we still have to configure the sensor over i2c
// If you don't do this, the strobe GPIO is an output (even in reset it seems!)
const CameraConfig WIDE_ROAD_CAMERA_CONFIG = {
  .camera_num = 0,
  .stream_type = VISION_STREAM_WIDE_ROAD,
  .focal_length_mm = 1.71,
  .publish_name = "wideRoadCameraState",
  .init_camera_state = &cereal::Event::Builder::initWideRoadCameraState,
  .enabled = !getenv("DISABLE_WIDE_ROAD"),
  .phy = CAM_ISP_IFE_IN_RES_PHY_0,
  .expo_rect = {96, 250, 1734, 524},
  .expo_focal_length = 567.0,
};

const CameraConfig ROAD_CAMERA_CONFIG = {
  .camera_num = 1,
  .stream_type = VISION_STREAM_ROAD,
  .focal_length_mm = 8.0,
  .publish_name = "roadCameraState",
  .init_camera_state = &cereal::Event::Builder::initRoadCameraState,
  .enabled = !getenv("DISABLE_ROAD"),
  .phy = CAM_ISP_IFE_IN_RES_PHY_1,
  .expo_rect = {96, 160, 1734, 986},
  .expo_focal_length = 2648.0,
};

const CameraConfig DRIVER_CAMERA_CONFIG = {
  .camera_num = 2,
  .stream_type = VISION_STREAM_DRIVER,
  .focal_length_mm = 1.71,
  .publish_name = "driverCameraState",
  .init_camera_state = &cereal::Event::Builder::initDriverCameraState,
  .enabled = !getenv("DISABLE_DRIVER"),
  .phy = CAM_ISP_IFE_IN_RES_PHY_2,
  .expo_rect = {96, 242, 1736, 906},
  .expo_focal_length = 567.0,
};
