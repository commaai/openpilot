#include "system/camerad/cameras/camera_server.h"

#include <fcntl.h>
#include <poll.h>
#include <sys/ioctl.h>

#include <cassert>

#include "common/clutil.h"
#include "media/cam_isp.h"
#ifdef QCOM2
#include "CL/cl_ext_qcom.h"
#endif

#include "common/swaglog.h"
#include "system/camerad/cameras/camera_common.h"

ExitHandler do_exit;

CameraServer::CameraServer() {
  device_id = cl_get_device_id(CL_DEVICE_TYPE_DEFAULT);
#ifdef QCOM2
  const cl_context_properties props[] = {CL_CONTEXT_PRIORITY_HINT_QCOM, CL_PRIORITY_HINT_HIGH_QCOM, 0};
  context.reset(CL_CHECK_ERR(clCreateContext(props, 1, &device_id, NULL, NULL, &err)));
#else
  context.reset(CL_CHECK_ERR(clCreateContext(NULL, 1, &device_id, NULL, NULL, &err)));
#endif

  vipc_server = std::make_unique<VisionIpcServer>("camerad", device_id, context.get());
  pm.reset(new PubMaster({"roadCameraState", "driverCameraState", "wideRoadCameraState", "thumbnail"}));

  LOG("-- Opening devices");
  // video0 is req_mgr, the target of many ioctls
  video0_fd = HANDLE_EINTR(open("/dev/v4l/by-path/platform-soc:qcom_cam-req-mgr-video-index0", O_RDWR | O_NONBLOCK));
  assert(video0_fd >= 0);
  LOGD("opened video0");

  // video1 is cam_sync, the target of some ioctls
  cam_sync_fd = HANDLE_EINTR(open("/dev/v4l/by-path/platform-cam_sync-video-index0", O_RDWR | O_NONBLOCK));
  assert(cam_sync_fd >= 0);
  LOGD("opened video1 (cam_sync)");

  // looks like there's only one of these
  isp_fd = open_v4l_by_name_and_index("cam-isp");
  assert(isp_fd >= 0);
  LOGD("opened isp");

  // query icp for MMU handles
  LOG("-- Query ICP for MMU handles");
  struct cam_isp_query_cap_cmd isp_query_cap_cmd = {0};
  struct cam_query_cap_cmd query_cap_cmd = {0};
  query_cap_cmd.handle_type = 1;
  query_cap_cmd.caps_handle = (uint64_t)&isp_query_cap_cmd;
  query_cap_cmd.size = sizeof(isp_query_cap_cmd);
  int ret = do_cam_control(isp_fd, CAM_QUERY_CAP, &query_cap_cmd, sizeof(query_cap_cmd));
  assert(ret == 0);
  LOGD("using MMU handle: %x", isp_query_cap_cmd.device_iommu.non_secure);
  LOGD("using MMU handle: %x", isp_query_cap_cmd.cdm_iommu.non_secure);
  device_iommu = isp_query_cap_cmd.device_iommu.non_secure;
  cdm_iommu = isp_query_cap_cmd.cdm_iommu.non_secure;

  // subscribe
  LOG("-- Subscribing");
  struct v4l2_event_subscription sub = {0};
  sub.type = V4L_EVENT_CAM_REQ_MGR_EVENT;
  sub.id = V4L_EVENT_CAM_REQ_MGR_SOF_BOOT_TS;
  ret = HANDLE_EINTR(ioctl(video0_fd, VIDIOC_SUBSCRIBE_EVENT, &sub));
  LOGD("req mgr subscribe: %d", ret);

  driver_cam.camera_open(this, 2, !env_disable_driver);
  LOGD("driver camera opened");
  road_cam.camera_open(this, 1, !env_disable_road);
  LOGD("road camera opened");
  wide_road_cam.camera_open(this, 0, !env_disable_wide_road);
  LOGD("wide road camera opened");

  driver_cam.camera_init(20, VISION_STREAM_DRIVER);
  road_cam.camera_init(20, VISION_STREAM_ROAD);
  wide_road_cam.camera_init(20, VISION_STREAM_WIDE_ROAD);
}

CameraServer::~CameraServer() {
  driver_cam.camera_close();
  road_cam.camera_close();
  wide_road_cam.camera_close();
}

void CameraServer::run() {
  LOG("-- Starting threads");
  std::vector<std::thread> threads;
  if (driver_cam.enabled) threads.emplace_back(&CameraState::frameThread, &driver_cam);
  if (road_cam.enabled) threads.emplace_back(&CameraState::frameThread, &road_cam);
  if (wide_road_cam.enabled) threads.emplace_back(&CameraState::frameThread, &wide_road_cam);

  // start devices
  LOG("-- Starting devices");
  driver_cam.sensors_start();
  road_cam.sensors_start();
  wide_road_cam.sensors_start();

  // poll events
  LOG("-- Dequeueing Video events");
  while (!do_exit) {
    struct pollfd fds[1] = {{0}};

    fds[0].fd = video0_fd;
    fds[0].events = POLLPRI;

    int ret = poll(fds, std::size(fds), 1000);
    if (ret < 0) {
      if (errno == EINTR || errno == EAGAIN) continue;
      LOGE("poll failed (%d - %d)", ret, errno);
      break;
    }

    if (!fds[0].revents) continue;

    struct v4l2_event ev = {0};
    ret = HANDLE_EINTR(ioctl(fds[0].fd, VIDIOC_DQEVENT, &ev));
    if (ret == 0) {
      if (ev.type == V4L_EVENT_CAM_REQ_MGR_EVENT) {
        struct cam_req_mgr_message *event_data = (struct cam_req_mgr_message *)ev.u.data;
        // LOGD("v4l2 event: sess_hdl 0x%X, link_hdl 0x%X, frame_id %d, req_id %lld, timestamp 0x%llx, sof_status %d\n", event_data->session_hdl, event_data->u.frame_msg.link_hdl, event_data->u.frame_msg.frame_id, event_data->u.frame_msg.request_id, event_data->u.frame_msg.timestamp, event_data->u.frame_msg.sof_status);
        if (env_debug_frames) {
          printf("sess_hdl 0x%6X, link_hdl 0x%6X, frame_id %lu, req_id %lu, timestamp %.2f ms, sof_status %d\n", event_data->session_hdl, event_data->u.frame_msg.link_hdl, event_data->u.frame_msg.frame_id, event_data->u.frame_msg.request_id, event_data->u.frame_msg.timestamp / 1e6, event_data->u.frame_msg.sof_status);
        }

        if (event_data->session_hdl == road_cam.session_handle) {
          road_cam.handle_camera_event(event_data);
        } else if (event_data->session_hdl == wide_road_cam.session_handle) {
          wide_road_cam.handle_camera_event(event_data);
        } else if (event_data->session_hdl == driver_cam.session_handle) {
          driver_cam.handle_camera_event(event_data);
        } else {
          LOGE("Unknown vidioc event source");
          assert(false);
        }
      }
    } else {
      LOGE("VIDIOC_DQEVENT failed, errno=%d", errno);
    }
  }

  LOG(" ************** STOPPING **************");
  for (auto &t : threads) t.join();
}
