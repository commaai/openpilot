#include "selfdrive/camerad/cameras/camera_qcom2.h"

#include <fcntl.h>
#include <poll.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>

#include <atomic>
#include <cassert>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstring>

#include "media/cam_defs.h"
#include "media/cam_isp.h"
#include "media/cam_isp_ife.h"
#include "media/cam_sensor.h"
#include "media/cam_sensor_cmn_header.h"
#include "media/cam_sync.h"
#include "selfdrive/common/swaglog.h"

extern ExitHandler do_exit;

// ******************* camera *******************

void cameras_init(VisionIpcServer *v, MultiCameraState *s, cl_device_id device_id, cl_context ctx) {
  s->driver_cam.camera_init(s, v, CAMERA_ID_AR0231, 2, 20, device_id, ctx, VISION_STREAM_RGB_FRONT, VISION_STREAM_DRIVER);
  printf("driver camera initted \n");
  if (!env_only_driver) {
    s->road_cam.camera_init(s, v, CAMERA_ID_AR0231, 1, 20, device_id, ctx, VISION_STREAM_RGB_BACK, VISION_STREAM_ROAD); // swap left/right
    printf("road camera initted \n");
    s->wide_road_cam.camera_init(s, v, CAMERA_ID_AR0231, 0, 20, device_id, ctx, VISION_STREAM_RGB_WIDE, VISION_STREAM_WIDE_ROAD);
    printf("wide road camera initted \n");
  }

  s->sm = new SubMaster({"driverState"});
  s->pm = new PubMaster({"roadCameraState", "driverCameraState", "wideRoadCameraState", "thumbnail"});
}

void cameras_open(MultiCameraState *s) {
  int ret;

  LOG("-- Opening devices");
  // video0 is req_mgr, the target of many ioctls
  s->video0_fd = HANDLE_EINTR(open("/dev/v4l/by-path/platform-soc:qcom_cam-req-mgr-video-index0", O_RDWR | O_NONBLOCK));
  assert(s->video0_fd >= 0);
  LOGD("opened video0");

  // video1 is cam_sync, the target of some ioctls
  s->video1_fd = HANDLE_EINTR(open("/dev/v4l/by-path/platform-cam_sync-video-index0", O_RDWR | O_NONBLOCK));
  assert(s->video1_fd >= 0);
  LOGD("opened video1");

  // looks like there's only one of these
  s->isp_fd = HANDLE_EINTR(open("/dev/v4l-subdev1", O_RDWR | O_NONBLOCK));
  assert(s->isp_fd >= 0);
  LOGD("opened isp");

  // query icp for MMU handles
  LOG("-- Query ICP for MMU handles");
  static struct cam_isp_query_cap_cmd isp_query_cap_cmd = {0};
  static struct cam_query_cap_cmd query_cap_cmd = {0};
  query_cap_cmd.handle_type = 1;
  query_cap_cmd.caps_handle = (uint64_t)&isp_query_cap_cmd;
  query_cap_cmd.size = sizeof(isp_query_cap_cmd);
  ret = do_cam_control(s->isp_fd, CAM_QUERY_CAP, &query_cap_cmd, sizeof(query_cap_cmd));
  assert(ret == 0);
  LOGD("using MMU handle: %x", isp_query_cap_cmd.device_iommu.non_secure);
  LOGD("using MMU handle: %x", isp_query_cap_cmd.cdm_iommu.non_secure);
  s->device_iommu = isp_query_cap_cmd.device_iommu.non_secure;
  s->cdm_iommu = isp_query_cap_cmd.cdm_iommu.non_secure;

  // subscribe
  LOG("-- Subscribing");
  static struct v4l2_event_subscription sub = {0};
  sub.type = V4L_EVENT_CAM_REQ_MGR_EVENT;
  sub.id = 2; // should use boot time for sof
  ret = HANDLE_EINTR(ioctl(s->video0_fd, VIDIOC_SUBSCRIBE_EVENT, &sub));
  printf("req mgr subscribe: %d\n", ret);

  s->driver_cam.camera_open();
  printf("driver camera opened \n");
  if (!env_only_driver) {
    s->road_cam.camera_open();
    printf("road camera opened \n");
    s->wide_road_cam.camera_open();
    printf("wide road camera opened \n");
  }
}

void cameras_close(MultiCameraState *s) {
  s->driver_cam.camera_close();
  if (!env_only_driver) {
    s->road_cam.camera_close();
    s->wide_road_cam.camera_close();
  }

  delete s->sm;
  delete s->pm;
}

// ******************* just a helper *******************


void camera_autoexposure(CameraState *s, float grey_frac) {
  s->set_camera_exposure(grey_frac);
}

// called by processing_thread
void process_road_camera(MultiCameraState *s, CameraState *c, int cnt) {
  const CameraBuf *b = &c->buf;

  MessageBuilder msg;
  auto framed = c == &s->road_cam ? msg.initEvent().initRoadCameraState() : msg.initEvent().initWideRoadCameraState();
  fill_frame_data(framed, b->cur_frame_data);
  if ((c == &s->road_cam && env_send_road) || (c == &s->wide_road_cam && env_send_wide_road)) {
    framed.setImage(get_frame_image(b));
  }
  if (c == &s->road_cam) {
    framed.setTransform(b->yuv_transform.v);
  }
  s->pm->send(c == &s->road_cam ? "roadCameraState" : "wideRoadCameraState", msg);

  const auto [x, y, w, h] = (c == &s->wide_road_cam) ? std::tuple(96, 250, 1734, 524) : std::tuple(96, 160, 1734, 986);
  const int skip = 2;
  camera_autoexposure(c, set_exposure_target(b, x, x + w, skip, y, y + h, skip));
}

void cameras_run(MultiCameraState *s) {
  LOG("-- Starting threads");
  std::vector<std::thread> threads;
  threads.push_back(start_process_thread(s, &s->driver_cam, common_process_driver_camera));
  if (!env_only_driver) {
    threads.push_back(start_process_thread(s, &s->road_cam, process_road_camera));
    threads.push_back(start_process_thread(s, &s->wide_road_cam, process_road_camera));
  }

  // start devices
  LOG("-- Starting devices");
  s->driver_cam.start();
  if (!env_only_driver) {
    s->road_cam.start();
    s->wide_road_cam.start();
  }

  // poll events
  LOG("-- Dequeueing Video events");
  while (!do_exit) {
    struct pollfd fds[1] = {{0}};

    fds[0].fd = s->video0_fd;
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
          printf("sess_hdl 0x%X, link_hdl 0x%X, frame_id %lu, req_id %lu, timestamp 0x%lx, sof_status %d\n", event_data->session_hdl, event_data->u.frame_msg.link_hdl, event_data->u.frame_msg.frame_id, event_data->u.frame_msg.request_id, event_data->u.frame_msg.timestamp, event_data->u.frame_msg.sof_status);
        }

        if (event_data->session_hdl == s->road_cam.session_handle) {
          s->road_cam.handle_camera_event(event_data);
        } else if (event_data->session_hdl == s->wide_road_cam.session_handle) {
          s->wide_road_cam.handle_camera_event(event_data);
        } else if (event_data->session_hdl == s->driver_cam.session_handle) {
          s->driver_cam.handle_camera_event(event_data);
        } else {
          printf("Unknown vidioc event source\n");
          assert(false);
        }
      }
    } else {
      LOGE("VIDIOC_DQEVENT failed, errno=%d", errno);
    }
  }

  LOG(" ************** STOPPING **************");

  for (auto &t : threads) t.join();

  cameras_close(s);
}
