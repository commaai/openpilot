#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <fcntl.h>
#include <sys/ioctl.h>

#include "common/util.h"
#include "common/swaglog.h"
#include "camera_qcom2.h"

#include "media/cam_defs.h"
#include "media/cam_isp.h"
#include "media/cam_isp_ife.h"
#include "media/cam_req_mgr.h"
#include "media/cam_sensor_cmn_header.h"
#include "media/cam_sensor.h"
#include "media/cam_sync.h"

#define FRAME_WIDTH  1928
#define FRAME_HEIGHT 1208
#define FRAME_STRIDE 1936

CameraInfo cameras_supported[CAMERA_ID_MAX] = {
  [CAMERA_ID_AR0231] = {
      .frame_width = FRAME_WIDTH,
      .frame_height = FRAME_HEIGHT,
      .frame_stride = FRAME_STRIDE,
      .bayer = false,
      .bayer_flip = false,
  },
};

// ************** camera helpers ****************

int cam_control(int fd, int op_code, void *handle, int size) {
  struct cam_control camcontrol = {0};
  camcontrol.op_code = op_code;
  camcontrol.handle = (uint64_t)handle;
  if (size == 0) { camcontrol.size = 8;
    camcontrol.handle_type = CAM_HANDLE_MEM_HANDLE;
  } else {
    camcontrol.size = size;
    camcontrol.handle_type = CAM_HANDLE_USER_POINTER;
  }

  int ret = ioctl(fd, VIDIOC_CAM_CONTROL, &camcontrol);
  if (ret == -1) {
    perror("wat");
  }
  return ret;
}


// ******************* camera *******************

static void camera_release_buffer(void* cookie, int buf_idx) {
  CameraState *s = cookie;

  // printf("camera_release_buffer %d\n", buf_idx);
  /*s->ss[0].qbuf_info[buf_idx].dirty_buf = 1;
  ioctl(s->isp_fd, VIDIOC_MSM_ISP_ENQUEUE_BUF, &s->ss[0].qbuf_info[buf_idx]);*/
}

static void camera_init(CameraState *s, int camera_id, int camera_num, unsigned int fps) {
  LOGD("camera init %d", camera_num);

  // TODO: this is copied code from camera_webcam
  assert(camera_id < ARRAYSIZE(cameras_supported));
  s->ci = cameras_supported[camera_id];
  assert(s->ci.frame_width != 0);

  s->camera_num = camera_num;
  s->frame_size = s->ci.frame_height * s->ci.frame_stride;

  tbuffer_init2(&s->camera_tb, FRAME_BUF_COUNT, "frame", camera_release_buffer, s);

  s->transform = (mat3){{
    1.0, 0.0, 0.0,
    0.0, 1.0, 0.0,
    0.0, 0.0, 1.0,
  }};
}

static void camera_open(CameraState *s, VisionBuf* b) {
  int ret;
  s->bufs = b;

  // /dev/v4l-subdev10 is sensor, 11, 12, 13 are the other sensors
  switch (s->camera_num) {
    case 0:
      s->sensor_fd = open("/dev/v4l-subdev10", O_RDWR | O_NONBLOCK);
      break;
    case 1:
      s->sensor_fd = open("/dev/v4l-subdev11", O_RDWR | O_NONBLOCK);
      break;
    case 2:
      s->sensor_fd = open("/dev/v4l-subdev12", O_RDWR | O_NONBLOCK);
      break;
  }
  assert(s->sensor_fd >= 0);
  LOGD("opened sensor");

  // also at /dev/v4l-subdev3, 4, 5, 6
  switch (s->camera_num) {
    case 0:
      s->csiphy_fd = open("/dev/v4l-subdev3", O_RDWR | O_NONBLOCK);
      break;
    case 1:
      s->csiphy_fd = open("/dev/v4l-subdev4", O_RDWR | O_NONBLOCK);
      break;
    case 2:
      s->csiphy_fd = open("/dev/v4l-subdev5", O_RDWR | O_NONBLOCK);
      break;
  }
  assert(s->csiphy_fd >= 0);
  LOGD("opened csiphy");

}

void cameras_init(DualCameraState *s) {
  camera_init(&s->rear, CAMERA_ID_AR0231, 0, 20);
  camera_init(&s->wide, CAMERA_ID_AR0231, 1, 20);
  camera_init(&s->front, CAMERA_ID_AR0231, 2, 20);
}

void cameras_open(DualCameraState *s, VisionBuf *camera_bufs_rear, VisionBuf *camera_bufs_focus, VisionBuf *camera_bufs_stats, VisionBuf *camera_bufs_front) {
  int ret;

  LOGD("\n-- Opening devices\n");
  // video0 is the target of many ioctls
  s->video0_fd = open("/dev/video0", O_RDWR | O_NONBLOCK);
  assert(s->video0_fd >= 0);
  LOGD("opened video0\n");
  s->rear.video0_fd = s->front.video0_fd = s->wide.video0_fd = s->video0_fd;

  // video1 is the target of some ioctls
  s->video1_fd = open("/dev/video1", O_RDWR | O_NONBLOCK);
  assert(s->video1_fd >= 0);
  LOGD("opened video1");
  s->rear.video1_fd = s->front.video1_fd = s->wide.video1_fd = s->video1_fd;

  s->isp_fd = open("/dev/v4l-subdev1", O_RDWR | O_NONBLOCK);
  assert(s->isp_fd >= 0);
  LOGD("opened isp");
  s->rear.isp_fd = s->front.isp_fd = s->wide.isp_fd = s->isp_fd;

  // query icp for MMU handles
  printf("\n-- Query ICP for MMU handles\n");
  static struct cam_isp_query_cap_cmd isp_query_cap_cmd = {0};
  static struct cam_query_cap_cmd query_cap_cmd = {0};
  query_cap_cmd.handle_type = 1;
  query_cap_cmd.caps_handle = (uint64_t)&isp_query_cap_cmd;
  query_cap_cmd.size = sizeof(isp_query_cap_cmd);
  ret = cam_control(s->isp_fd, CAM_QUERY_CAP, &query_cap_cmd, sizeof(query_cap_cmd));
  assert(ret == 0);
  LOGD("using MMU handle: %x", isp_query_cap_cmd.device_iommu.non_secure);
  LOGD("using MMU handle: %x", isp_query_cap_cmd.cdm_iommu.non_secure);
  int device_iommu = isp_query_cap_cmd.device_iommu.non_secure;
  int cdm_iommu = isp_query_cap_cmd.cdm_iommu.non_secure;
  s->rear.device_iommu = s->front.device_iommu = s->wide.device_iommu = device_iommu;
  s->rear.cdm_iommu = s->front.cdm_iommu = s->wide.cdm_iommu = cdm_iommu;

  camera_open(&s->rear, camera_bufs_rear);
  camera_open(&s->front, camera_bufs_front);
  // TODO: add bufs for camera wide
}

static void cameras_close(DualCameraState *s) {
}

void cameras_run(DualCameraState *s) {
  // TODO: loop
  LOG(" ************** STOPPING **************");
  cameras_close(s);
}

void camera_autoexposure(CameraState *s, float grey_frac) {
}

