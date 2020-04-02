#include "camera_webcam.h"

#include <unistd.h>
#include <string.h>
#include <signal.h>
#include <pthread.h>

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
#define FRAME_WIDTH_FRONT  1152
#define FRAME_HEIGHT_FRONT 864

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

static void* rear_thread(void *arg) {
  int err;

  set_thread_name("webcam_rear_thread");
  CameraState* s = (CameraState*)arg;

  cv::VideoCapture cap_rear(1); // road
  cap_rear.set(cv::CAP_PROP_FRAME_WIDTH, 853);
  cap_rear.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
  cap_rear.set(cv::CAP_PROP_FPS, s->fps);
  cap_rear.set(cv::CAP_PROP_AUTOFOCUS, 0); // off
  cap_rear.set(cv::CAP_PROP_FOCUS, 0); // 0 - 255?
  // cv::Rect roi_rear(160, 0, 960, 720);

  cv::Size size;
  size.height = s->ci.frame_height;
  size.width = s->ci.frame_width;

  // transforms calculation see tools/webcam/warp_vis.py
  float ts[9] = {1.50330396, 0.0, -59.40969163,
                  0.0, 1.50330396, 76.20704846,
                  0.0, 0.0, 1.0};
  // if camera upside down:
  // float ts[9] = {-1.50330396, 0.0, 1223.4,
  //                 0.0, -1.50330396, 797.8,
  //                 0.0, 0.0, 1.0};
  const cv::Mat transform = cv::Mat(3, 3, CV_32F, ts);

  if (!cap_rear.isOpened()) {
    err = 1;
  }

  uint32_t frame_id = 0;
  TBuffer* tb = &s->camera_tb;

  while (!do_exit) {
    cv::Mat frame_mat;
    cv::Mat transformed_mat;

    cap_rear >> frame_mat;

    // int rows = frame_mat.rows;
    // int cols = frame_mat.cols;
    // printf("Raw Rear, R=%d, C=%d\n", rows, cols);

    cv::warpPerspective(frame_mat, transformed_mat, transform, size, cv::INTER_LINEAR, cv::BORDER_CONSTANT, 0);

    int transformed_size = transformed_mat.total() * transformed_mat.elemSize();

    const int buf_idx = tbuffer_select(tb);
    s->camera_bufs_metadata[buf_idx] = {
      .frame_id = frame_id,
    };

    cl_command_queue q = s->camera_bufs[buf_idx].copy_q;
    cl_mem yuv_cl = s->camera_bufs[buf_idx].buf_cl;
    cl_event map_event;
    void *yuv_buf = (void *)clEnqueueMapBuffer(q, yuv_cl, CL_TRUE,
                                                CL_MAP_WRITE, 0, transformed_size,
                                                0, NULL, &map_event, &err);
    assert(err == 0);
    clWaitForEvents(1, &map_event);
    clReleaseEvent(map_event);
    memcpy(yuv_buf, transformed_mat.data, transformed_size);

    clEnqueueUnmapMemObject(q, yuv_cl, yuv_buf, 0, NULL, &map_event);
    clWaitForEvents(1, &map_event);
    clReleaseEvent(map_event);
    tbuffer_dispatch(tb, buf_idx);

    frame_id += 1;
    frame_mat.release();
    transformed_mat.release();
  }

  cap_rear.release();
  return NULL;
}

void front_thread(CameraState *s) {
  int err;

  cv::VideoCapture cap_front(2); // driver
  cap_front.set(cv::CAP_PROP_FRAME_WIDTH, 853);
  cap_front.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
  cap_front.set(cv::CAP_PROP_FPS, s->fps);
  // cv::Rect roi_front(320, 0, 960, 720);

  cv::Size size;
  size.height = s->ci.frame_height;
  size.width = s->ci.frame_width;

  // transforms calculation see tools/webcam/warp_vis.py
  float ts[9] = {1.42070485, 0.0, -30.16740088,
                  0.0, 1.42070485, 91.030837,
                  0.0, 0.0, 1.0};
  // if camera upside down:
  // float ts[9] = {-1.42070485, 0.0, 1182.2,
  //                 0.0, -1.42070485, 773.0,
  //                 0.0, 0.0, 1.0};
  const cv::Mat transform = cv::Mat(3, 3, CV_32F, ts);

  if (!cap_front.isOpened()) {
    err = 1;
  }

  uint32_t frame_id = 0;
  TBuffer* tb = &s->camera_tb;

  while (!do_exit) {
    cv::Mat frame_mat;
    cv::Mat transformed_mat;

    cap_front >> frame_mat;

    // int rows = frame_mat.rows;
    // int cols = frame_mat.cols;
    // printf("Raw Front, R=%d, C=%d\n", rows, cols);

    cv::warpPerspective(frame_mat, transformed_mat, transform, size, cv::INTER_LINEAR, cv::BORDER_CONSTANT, 0);

    int transformed_size = transformed_mat.total() * transformed_mat.elemSize();

    const int buf_idx = tbuffer_select(tb);
    s->camera_bufs_metadata[buf_idx] = {
      .frame_id = frame_id,
    };

    cl_command_queue q = s->camera_bufs[buf_idx].copy_q;
    cl_mem yuv_cl = s->camera_bufs[buf_idx].buf_cl;
    cl_event map_event;
    void *yuv_buf = (void *)clEnqueueMapBuffer(q, yuv_cl, CL_TRUE,
                                                CL_MAP_WRITE, 0, transformed_size,
                                                0, NULL, &map_event, &err);
    assert(err == 0);
    clWaitForEvents(1, &map_event);
    clReleaseEvent(map_event);
    memcpy(yuv_buf, transformed_mat.data, transformed_size);

    clEnqueueUnmapMemObject(q, yuv_cl, yuv_buf, 0, NULL, &map_event);
    clWaitForEvents(1, &map_event);
    clReleaseEvent(map_event);
    tbuffer_dispatch(tb, buf_idx);

    frame_id += 1;
    frame_mat.release();
    transformed_mat.release();
  }

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
  [CAMERA_ID_LGC615] = {
      .frame_width = FRAME_WIDTH_FRONT,
      .frame_height = FRAME_HEIGHT_FRONT,
      .frame_stride = FRAME_WIDTH_FRONT*3,
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

  camera_init(&s->front, CAMERA_ID_LGC615, 10);
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

  int err;
  pthread_t rear_thread_handle;
  err = pthread_create(&rear_thread_handle, NULL,
                        rear_thread, &s->rear);
  assert(err == 0);

  front_thread(&s->front);

  err = pthread_join(rear_thread_handle, NULL);
  assert(err == 0);
  cameras_close(s);
}
