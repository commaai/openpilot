#pragma once

#include "common/util.h"
#include "cereal/gen/cpp/log.capnp.h"
#include "msgq/visionipc/visionipc_server.h"

#include "media/cam_isp_ife.h"


typedef enum {
  ISP_RAW_OUTPUT,   // raw frame from sensor
  ISP_IFE_PROCESSED,  // fully processed image through the IFE
  ISP_BPS_PROCESSED,  // fully processed image through the BPS
} SpectraOutputType;

// For the comma 3/3X three camera platform

struct CameraConfig {
  int camera_num;
  VisionStreamType stream_type;
  float focal_len;  // millimeters
  const char *publish_name;
  cereal::FrameData::Builder (cereal::Event::Builder::*init_camera_state)();
  bool enabled;
  uint32_t phy;
  bool vignetting_correction;
  SpectraOutputType output_type;
};

// NOTE: to be able to disable road and wide road, we still have to configure the sensor over i2c
// If you don't do this, the strobe GPIO is an output (even in reset it seems!)
const CameraConfig WIDE_ROAD_CAMERA_CONFIG = {
  .camera_num = 0,
  .stream_type = VISION_STREAM_WIDE_ROAD,
  .focal_len = 1.71,
  .publish_name = "wideRoadCameraState",
  .init_camera_state = &cereal::Event::Builder::initWideRoadCameraState,
  .enabled = !getenv("DISABLE_WIDE_ROAD"),
  .phy = CAM_ISP_IFE_IN_RES_PHY_0,
  .vignetting_correction = false,
  .output_type = ISP_IFE_PROCESSED,
};

const CameraConfig ROAD_CAMERA_CONFIG = {
  .camera_num = 1,
  .stream_type = VISION_STREAM_ROAD,
  .focal_len = 8.0,
  .publish_name = "roadCameraState",
  .init_camera_state = &cereal::Event::Builder::initRoadCameraState,
  .enabled = !getenv("DISABLE_ROAD"),
  .phy = CAM_ISP_IFE_IN_RES_PHY_1,
  .vignetting_correction = true,
  .output_type = ISP_IFE_PROCESSED,
};

const CameraConfig DRIVER_CAMERA_CONFIG = {
  .camera_num = 2,
  .stream_type = VISION_STREAM_DRIVER,
  .focal_len = 1.71,
  .publish_name = "driverCameraState",
  .init_camera_state = &cereal::Event::Builder::initDriverCameraState,
  .enabled = !getenv("DISABLE_DRIVER"),
  .phy = CAM_ISP_IFE_IN_RES_PHY_2,
  .vignetting_correction = false,
  .output_type = ISP_BPS_PROCESSED,
};

const CameraConfig ALL_CAMERA_CONFIGS[] = {WIDE_ROAD_CAMERA_CONFIG, ROAD_CAMERA_CONFIG, DRIVER_CAMERA_CONFIG};
