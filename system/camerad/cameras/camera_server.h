#pragma once

#include <media/cam_req_mgr.h>

#include <memory>

#include "cereal/visionipc/visionbuf.h"
#include "cereal/visionipc/visionipc.h"
#include "cereal/visionipc/visionipc_server.h"
#include "common/util.h"
#include "system/camerad/cameras/camera_qcom2.h"

class CameraServer {
public:
  CameraServer();
  ~CameraServer();
  void run();

  struct CLContextDeleter {
    void operator()(_cl_context *ctx) const { clReleaseContext(ctx); }
  };
  cl_device_id device_id;
  std::unique_ptr<_cl_context, CLContextDeleter> context;
  std::unique_ptr<VisionIpcServer> vipc_server;
  std::unique_ptr<PubMaster> pm;

  unique_fd video0_fd;
  unique_fd cam_sync_fd;
  unique_fd isp_fd;
  int device_iommu;
  int cdm_iommu;

  CameraState road_cam;
  CameraState wide_road_cam;
  CameraState driver_cam;
};
