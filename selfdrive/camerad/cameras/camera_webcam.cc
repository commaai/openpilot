#include "camera_webcam.h"

#include <unistd.h>
#include <assert.h>
#include <string.h>
#include <signal.h>
#include <pthread.h>

#include "common/util.h"
#include "common/timing.h"
#include "common/swaglog.h"

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wundefined-inline"
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#pragma clang diagnostic pop


extern volatile sig_atomic_t do_exit;

#define FRAME_WIDTH  1164
#define FRAME_HEIGHT 874
#define FRAME_WIDTH_FRONT  1152
#define FRAME_HEIGHT_FRONT 864

namespace {
void camera_open(CameraState *s, bool rear) {
}

void camera_close(CameraState *s) {
  s->buf.stop();
}

void camera_init(CameraState *s, int camera_id, unsigned int fps, cl_device_id device_id, cl_context ctx) {
  assert(camera_id < ARRAYSIZE(cameras_supported));
  s->ci = cameras_supported[camera_id];
  assert(s->ci.frame_width != 0);

  s->fps = fps;

  s->buf.init(device_id, ctx, s, FRAME_BUF_COUNT, "frame");
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
  TBuffer* tb = &s->buf.camera_tb;

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
    s->buf.camera_bufs_metadata[buf_idx] = {
      .frame_id = frame_id,
    };

    cl_command_queue q = s->buf.camera_bufs[buf_idx].copy_q;
    cl_mem yuv_cl = s->buf.camera_bufs[buf_idx].buf_cl;
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
  TBuffer* tb = &s->buf.camera_tb;

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
    s->buf.camera_bufs_metadata[buf_idx] = {
      .frame_id = frame_id,
    };

    cl_command_queue q = s->buf.camera_bufs[buf_idx].copy_q;
    cl_mem yuv_cl = s->buf.camera_bufs[buf_idx].buf_cl;
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

void cameras_init(MultiCameraState *s, cl_device_id device_id, cl_context ctx) {

  camera_init(&s->rear, CAMERA_ID_LGC920, 20, device_id, ctx);
  s->rear.transform = (mat3){{
    1.0, 0.0, 0.0,
    0.0, 1.0, 0.0,
    0.0, 0.0, 1.0,
  }};

  camera_init(&s->front, CAMERA_ID_LGC615, 10, device_id, ctx);
  s->front.transform = (mat3){{
    1.0, 0.0, 0.0,
    0.0, 1.0, 0.0,
    0.0, 0.0, 1.0,
  }};

  s->pm = new PubMaster({"frame", "frontFrame"});
}

void camera_autoexposure(CameraState *s, float grey_frac) {}

void cameras_open(MultiCameraState *s) {
  // LOG("*** open front ***");
  camera_open(&s->front, false);

  // LOG("*** open rear ***");
  camera_open(&s->rear, true);
}

void cameras_close(MultiCameraState *s) {
  camera_close(&s->rear);
  camera_close(&s->front);
  delete s->pm;
}

void camera_process_front(MultiCameraState *s, CameraState *c, int cnt) {
  MessageBuilder msg;
  auto framed = msg.initEvent().initFrontFrame();
  framed.setFrameType(cereal::FrameData::FrameType::FRONT);
  fill_frame_data(framed, c->buf.cur_frame_data, cnt);
  s->pm->send("frontFrame", msg);
}

void camera_process_rear(MultiCameraState *s, CameraState *c, int cnt) {
  const CameraBuf *b = &c->buf;
  MessageBuilder msg;
  auto framed = msg.initEvent().initFrame();
  fill_frame_data(framed, b->cur_frame_data, cnt);
  framed.setImage(kj::arrayPtr((const uint8_t *)b->yuv_ion[b->cur_yuv_idx].addr, b->yuv_buf_size));
  framed.setTransform(kj::ArrayPtr<const float>(&b->yuv_transform.v[0], 9));
  s->pm->send("frame", msg);
}

void cameras_run(MultiCameraState *s) {
  std::vector<std::thread> threads;
  threads.push_back(start_process_thread(s, "processing", &s->rear, 51, camera_process_rear));
  threads.push_back(start_process_thread(s, "frontview", &s->front, 51, camera_process_front));
  
  std::thread t_rear = std::thread(rear_thread, &s->rear);
  set_thread_name("webcam_thread");
  front_thread(&s->front);
  t_rear.join();
  cameras_close(s);
  
  for (auto &t : threads) t.join();
}
