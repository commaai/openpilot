#include "selfdrive/camerad/cameras/camera_webcam.h"

#include <unistd.h>

#include <cassert>
#include <cstring>

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wundefined-inline"
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/videoio.hpp>
#pragma clang diagnostic pop

#include "selfdrive/common/clutil.h"
#include "selfdrive/common/swaglog.h"
#include "selfdrive/common/timing.h"
#include "selfdrive/common/util.h"

// id of the video capturing device
const int ROAD_CAMERA_ID = util::getenv("ROADCAM_ID", 1);
const int DRIVER_CAMERA_ID = util::getenv("DRIVERCAM_ID", 2);

#define FRAME_WIDTH  1164
#define FRAME_HEIGHT 874
#define FRAME_WIDTH_FRONT  1152
#define FRAME_HEIGHT_FRONT 864

extern ExitHandler do_exit;

namespace {

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

void camera_open(CameraState *s, bool rear) {
  // empty
}

void camera_close(CameraState *s) {
  // empty
}

void camera_init(VisionIpcServer * v, CameraState *s, int camera_id, unsigned int fps, cl_device_id device_id, cl_context ctx, VisionStreamType rgb_type, VisionStreamType yuv_type) {
  assert(camera_id < std::size(cameras_supported));
  s->ci = cameras_supported[camera_id];
  assert(s->ci.frame_width != 0);

  s->camera_num = camera_id;
  s->fps = fps;
  s->buf.init(device_id, ctx, s, v, FRAME_BUF_COUNT, rgb_type, yuv_type);
}

void run_camera(CameraState *s, cv::VideoCapture &video_cap, float *ts) {
  assert(video_cap.isOpened());

  cv::Size size(s->ci.frame_width, s->ci.frame_height);
  const cv::Mat transform = cv::Mat(3, 3, CV_32F, ts);
  uint32_t frame_id = 0;
  size_t buf_idx = 0;

  while (!do_exit) {
    cv::Mat frame_mat, transformed_mat;
    video_cap >> frame_mat;
    if (frame_mat.empty()) continue;

    cv::warpPerspective(frame_mat, transformed_mat, transform, size, cv::INTER_LINEAR, cv::BORDER_CONSTANT, 0);

    s->buf.camera_bufs_metadata[buf_idx] = {.frame_id = frame_id};

    auto &buf = s->buf.camera_bufs[buf_idx];
    int transformed_size = transformed_mat.total() * transformed_mat.elemSize();
    CL_CHECK(clEnqueueWriteBuffer(buf.copy_q, buf.buf_cl, CL_TRUE, 0, transformed_size, transformed_mat.data, 0, NULL, NULL));

    s->buf.queue(buf_idx);

    ++frame_id;
    buf_idx = (buf_idx + 1) % FRAME_BUF_COUNT;
  }
}

static void road_camera_thread(CameraState *s) {
  set_thread_name("webcam_road_camera_thread");

  cv::VideoCapture cap_road(ROAD_CAMERA_ID, cv::CAP_V4L2); // road
  cap_road.set(cv::CAP_PROP_FRAME_WIDTH, 853);
  cap_road.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
  cap_road.set(cv::CAP_PROP_FPS, s->fps);
  cap_road.set(cv::CAP_PROP_AUTOFOCUS, 0); // off
  cap_road.set(cv::CAP_PROP_FOCUS, 0); // 0 - 255?
  // cv::Rect roi_rear(160, 0, 960, 720);

  // transforms calculation see tools/webcam/warp_vis.py
  float ts[9] = {1.50330396, 0.0, -59.40969163,
                  0.0, 1.50330396, 76.20704846,
                  0.0, 0.0, 1.0};
  // if camera upside down:
  // float ts[9] = {-1.50330396, 0.0, 1223.4,
  //                 0.0, -1.50330396, 797.8,
  //                 0.0, 0.0, 1.0};

  run_camera(s, cap_road, ts);
}

void driver_camera_thread(CameraState *s) {
  cv::VideoCapture cap_driver(DRIVER_CAMERA_ID, cv::CAP_V4L2); // driver
  cap_driver.set(cv::CAP_PROP_FRAME_WIDTH, 853);
  cap_driver.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
  cap_driver.set(cv::CAP_PROP_FPS, s->fps);
  // cv::Rect roi_front(320, 0, 960, 720);

  // transforms calculation see tools/webcam/warp_vis.py
  float ts[9] = {1.42070485, 0.0, -30.16740088,
                  0.0, 1.42070485, 91.030837,
                  0.0, 0.0, 1.0};
  // if camera upside down:
  // float ts[9] = {-1.42070485, 0.0, 1182.2,
  //                 0.0, -1.42070485, 773.0,
  //                 0.0, 0.0, 1.0};
  run_camera(s, cap_driver, ts);
}

}  // namespace

void cameras_init(VisionIpcServer *v, MultiCameraState *s, cl_device_id device_id, cl_context ctx) {
  camera_init(v, &s->road_cam, CAMERA_ID_LGC920, 20, device_id, ctx,
              VISION_STREAM_RGB_BACK, VISION_STREAM_ROAD);
  camera_init(v, &s->driver_cam, CAMERA_ID_LGC615, 10, device_id, ctx,
              VISION_STREAM_RGB_FRONT, VISION_STREAM_DRIVER);
  s->pm = new PubMaster({"roadCameraState", "driverCameraState", "thumbnail"});
}

void camera_autoexposure(CameraState *s, float grey_frac) {}

void cameras_open(MultiCameraState *s) {
  // LOG("*** open driver camera ***");
  camera_open(&s->driver_cam, false);
  // LOG("*** open road camera ***");
  camera_open(&s->road_cam, true);
}

void cameras_close(MultiCameraState *s) {
  camera_close(&s->road_cam);
  camera_close(&s->driver_cam);
  delete s->pm;
}

void process_driver_camera(MultiCameraState *s, CameraState *c, int cnt) {
  MessageBuilder msg;
  auto framed = msg.initEvent().initDriverCameraState();
  framed.setFrameType(cereal::FrameData::FrameType::FRONT);
  fill_frame_data(framed, c->buf.cur_frame_data);
  s->pm->send("driverCameraState", msg);
}

void process_road_camera(MultiCameraState *s, CameraState *c, int cnt) {
  const CameraBuf *b = &c->buf;
  MessageBuilder msg;
  auto framed = msg.initEvent().initRoadCameraState();
  fill_frame_data(framed, b->cur_frame_data);
  if (env_send_road) {
    framed.setImage(get_frame_image(&c->buf));
  }
  framed.setTransform(b->yuv_transform.v);
  s->pm->send("roadCameraState", msg);
}

void cameras_run(MultiCameraState *s) {
  std::vector<std::thread> threads;
  threads.push_back(start_process_thread(s, &s->road_cam, process_road_camera));
  threads.push_back(start_process_thread(s, &s->driver_cam, process_driver_camera));

  std::thread t_rear = std::thread(road_camera_thread, &s->road_cam);
  set_thread_name("webcam_thread");
  driver_camera_thread(&s->driver_cam);

  t_rear.join();

  for (auto &t : threads) t.join();

  cameras_close(s);
}
