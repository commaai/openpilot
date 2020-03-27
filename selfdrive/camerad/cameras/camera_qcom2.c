#include <assert.h>
#include <sys/ioctl.h>

#include "common/util.h"
#include "common/swaglog.h"
#include "camera_qcom2.h"

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

void cameras_init(DualCameraState *s) {
  camera_init(&s->rear, CAMERA_ID_AR0231, 0, 20);
  camera_init(&s->wide, CAMERA_ID_AR0231, 1, 20);
  camera_init(&s->front, CAMERA_ID_AR0231, 2, 20);
}

void cameras_open(DualCameraState *s, VisionBuf *camera_bufs_rear, VisionBuf *camera_bufs_focus, VisionBuf *camera_bufs_stats, VisionBuf *camera_bufs_front) {
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

