#include "camera_webcam.h"

#include <unistd.h>
#include <string.h>
#include <signal.h>

#include "common/util.h"
#include "common/timing.h"
#include "common/swaglog.h"
#include "buffering.h"

#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>

extern volatile sig_atomic_t do_exit;

#define FRAME_WIDTH  1164
#define FRAME_HEIGHT 874

namespace {
void camera_open(CameraState *s, VisionBuf *camera_bufs, bool rear) {
  assert(camera_bufs);
  s->camera_bufs = camera_bufs;
}

void camera_close(CameraState *s) {
  tbuffer_stop(&s->camera_tb);
}

void camera_release_buffer(void *cookie, int buf_idx) {
  CameraState *s = static_cast<CameraState *>(cookie);
}

void camera_init(CameraState *s, int camera_id, unsigned int fps) {
  assert(camera_id < ARRAYSIZE(cameras_supported));
  s->ci = cameras_supported[camera_id];
  assert(s->ci.frame_width != 0);

  s->frame_size = s->ci.frame_height * s->ci.frame_stride;
  s->fps = fps;

  tbuffer_init2(&s->camera_tb, FRAME_BUF_COUNT, "frame", camera_release_buffer, s);
}

void run_webcam(DualCameraState *s) {
  int err;

  CameraState* cameras[2] = {&s->rear, &s->front};

  cv::VideoCapture cap_rear(0); // road
  cap_rear.set(cv::CAP_PROP_FRAME_WIDTH, cameras[0]->ci.frame_width);
  cap_rear.set(cv::CAP_PROP_FRAME_HEIGHT, cameras[0]->ci.frame_height);
  cap_rear.set(cv::CAP_PROP_FPS, cameras[0]->fps);
  cap_rear.set(cv::CAP_PROP_FOCUS, 0);

  cv::VideoCapture cap_front(1); // driver
  cap_front.set(cv::CAP_PROP_FRAME_WIDTH, cameras[1]->ci.frame_width);
  cap_front.set(cv::CAP_PROP_FRAME_HEIGHT, cameras[1]->ci.frame_height);
  cap_front.set(cv::CAP_PROP_FPS, cameras[1]->fps);

  if (!cap_rear.isOpened() || !cap_front.isOpened()) {
    err = 1;
  }

  cv::VideoCapture* vcap[2] = {&cap_rear, &cap_front};
  uint32_t frame_id[2] = {0, 0};

  while (!do_exit) {
    for (int i=0; i<2; i++) {
      cv::Mat frame_mat;

      (*vcap[i]) >> frame_mat;

      frame_id[i] += 1;

      //if (i==1) {cv::resize(frame_mat, frame_mat, cv::Size(480, 360));}

      int frame_size = frame_mat.total() * frame_mat.elemSize();

      // printf("C%d: %d,%d\n", i+1, frame_id[i], frame_size);

      auto *tb = &cameras[i]->camera_tb;
      const int buf_idx = tbuffer_select(tb);
      cameras[i]->camera_bufs_metadata[buf_idx] = {
        .frame_id = frame_id[i],
      };

      cl_command_queue q = cameras[i]->camera_bufs[buf_idx].copy_q;
      cl_mem yuv_cl = cameras[i]->camera_bufs[buf_idx].buf_cl;
      cl_event map_event;
      void *yuv_buf = (void *)clEnqueueMapBuffer(q, yuv_cl, CL_TRUE,
                                                  CL_MAP_WRITE, 0, frame_size,
                                                  0, NULL, &map_event, &err);
      assert(err == 0);
      clWaitForEvents(1, &map_event);
      clReleaseEvent(map_event);
      memcpy(yuv_buf, frame_mat.data, frame_size);

      clEnqueueUnmapMemObject(q, yuv_cl, yuv_buf, 0, NULL, &map_event);
      clWaitForEvents(1, &map_event);
      clReleaseEvent(map_event);
      tbuffer_dispatch(tb, buf_idx);

      frame_mat.release();
    }
  }

  cap_rear.release();
  cap_front.release();
  return;
}

}  // namespace

CameraInfo cameras_supported[CAMERA_ID_MAX] = {
  // road facing
  [CAMERA_ID_LGC920] = {
      .frame_width = FRAME_WIDTH,
      .frame_height = FRAME_HEIGHT,
      .frame_stride = FRAME_WIDTH*3,
      .bayer = false,
      .bayer_flip = false,
  },
  // driver facing
  [CAMERA_ID_LGC270] = {
      .frame_width = 960,
      .frame_height = 720,
      .frame_stride = 960*3,
      .bayer = false,
      .bayer_flip = false,
  },
};

void cameras_init(DualCameraState *s) {
  memset(s, 0, sizeof(*s));

  camera_init(&s->rear, CAMERA_ID_LGC920, 20);
  s->rear.transform = (mat3){{
    1.0, 0.0, 0.0,
    0.0, 1.0, 0.0,
    0.0, 0.0, 1.0,
  }};

  camera_init(&s->front, CAMERA_ID_LGC270, 10);
  s->front.transform = (mat3){{
    1.0, 0.0, 0.0,
    0.0, 1.0, 0.0,
    0.0, 0.0, 1.0,
  }};
}

void camera_autoexposure(CameraState *s, float grey_frac) {}

void cameras_open(DualCameraState *s, VisionBuf *camera_bufs_rear,
                  VisionBuf *camera_bufs_focus, VisionBuf *camera_bufs_stats,
                  VisionBuf *camera_bufs_front) {
  assert(camera_bufs_rear);
  assert(camera_bufs_front);
  int err;

  // LOG("*** open front ***");
  camera_open(&s->front, camera_bufs_front, false);

  // LOG("*** open rear ***");
  camera_open(&s->rear, camera_bufs_rear, true);
}

void cameras_close(DualCameraState *s) {
  camera_close(&s->rear);
  camera_close(&s->front);
}

void cameras_run(DualCameraState *s) {
  set_thread_name("webcam_thread");
  run_webcam(s);
  cameras_close(s);
}