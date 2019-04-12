#include "camera_fake.h"

#include <cstring>
#include <unistd.h>
#include <vector>

#include <czmq.h>
#include <libyuv.h>
#include <capnp/serialize.h>
#include "cereal/gen/cpp/log.capnp.h"

#include "common/util.h"
#include "common/timing.h"
#include "common/swaglog.h"
#include "buffering.h"

extern volatile int do_exit;

#define FRAME_WIDTH 1164
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

  tbuffer_init2(&s->camera_tb, FRAME_BUF_COUNT, "frame", camera_release_buffer,
                s);
}


void run_simulator(DualCameraState *s) {
  int err = 0;

  zsock_t *frame_sock = zsock_new_sub(">tcp://127.0.0.1:9003", "");
  assert(frame_sock);
  void *frame_sock_raw = zsock_resolve(frame_sock);

  CameraState *const rear_camera = &s->rear;

  auto *tb = &rear_camera->camera_tb;

  while (!do_exit) {
    const int buf_idx = tbuffer_select(tb);

    auto *buf = &rear_camera->camera_bufs[buf_idx];

    zmq_msg_t t_msg;
    err = zmq_msg_init(&t_msg);
    assert(err == 0);

    zmq_msg_t frame_msg;
    err = zmq_msg_init(&frame_msg);
    assert(err == 0);

    // recv multipart (t, frame)
    err = zmq_msg_recv(&t_msg, frame_sock_raw, 0);
    assert(err != -1);
    err = zmq_msg_recv(&frame_msg, frame_sock_raw, 0);
    assert(err != -1);

    assert(zmq_msg_size(&t_msg) >= 8);
    uint8_t* dat = (uint8_t*)zmq_msg_data(&t_msg);
    float t = *(float*)&dat[0];
    uint32_t frame = *(uint32_t*)&dat[4];

    rear_camera->camera_bufs_metadata[buf_idx] = {
        .frame_id = frame,
        .timestamp_eof = nanos_since_boot(),
        .frame_length = 0,
        .integ_lines = 0,
        .global_gain = 0,
    };


    assert(zmq_msg_size(&frame_msg) == rear_camera->frame_size);

    err = libyuv::RAWToRGB24((const uint8_t*)zmq_msg_data(&frame_msg), rear_camera->ci.frame_width*3,
               (uint8_t*)buf->addr, rear_camera->ci.frame_stride,
               rear_camera->ci.frame_width, rear_camera->ci.frame_height);
    assert(err == 0);

    visionbuf_sync(buf, VISIONBUF_SYNC_TO_DEVICE);
    tbuffer_dispatch(tb, buf_idx);

    err = zmq_msg_close(&frame_msg);
    assert(err == 0);
    err = zmq_msg_close(&t_msg);
    assert(err == 0);
  }

  zsock_destroy(&frame_sock);
}

void run_unlogger(DualCameraState *s) {
  zsock_t *frame_sock = zsock_new_sub(NULL, "");
  assert(frame_sock);
  int err = zsock_connect(frame_sock,
                          "ipc:///tmp/9464f05d-9d88-4fc9-aa17-c75352d9590d");
  assert(err == 0);
  void *frame_sock_raw = zsock_resolve(frame_sock);

  CameraState *const rear_camera = &s->rear;
  auto frame_data = std::vector<capnp::word>{};

  auto *tb = &rear_camera->camera_tb;

  while (!do_exit) {
    // Handle rear camera only.
    zmq_msg_t msg;
    int rc = zmq_msg_init(&msg);
    assert(rc == 0);
    rc = zmq_msg_recv(&msg, frame_sock_raw, 0);
    if (rc == -1) {
      if (do_exit) {
        break;
      } else {
        fprintf(stderr, "Could not recv frame message: %d\n", errno);
      }
    }
    assert(rc != -1);

    const size_t msg_size_words = zmq_msg_size(&msg) / sizeof(capnp::word);
    assert(msg_size_words * sizeof(capnp::word) == zmq_msg_size(&msg));

    if (frame_data.size() < msg_size_words) {
      frame_data = std::vector<capnp::word>{msg_size_words};
    }
    std::memcpy(frame_data.data(), zmq_msg_data(&msg),
                msg_size_words * sizeof(*frame_data.data()));
    zmq_msg_close(&msg);

    capnp::FlatArrayMessageReader message{
        kj::arrayPtr(frame_data.data(), msg_size_words), {}};

    const auto &event = message.getRoot<cereal::Event>();
    assert(event.which() == cereal::Event::FRAME);
    const auto reader = event.getFrame();
    assert(reader.hasImage());
    const auto yuv_image = reader.getImage();

    // Copy camera data to buffer.
    const size_t width = rear_camera->ci.frame_width;
    const size_t height = rear_camera->ci.frame_height;

    const size_t y_len = width * height;
    const uint8_t *const y = yuv_image.begin();
    const uint8_t *const u = y + y_len;
    const uint8_t *const v = u + y_len / 4;

    assert(yuv_image.size() == y_len * 3 / 2);

    const int buf_idx = tbuffer_select(tb);
    rear_camera->camera_bufs_metadata[buf_idx] = {
        .frame_id = reader.getFrameId(),
        .timestamp_eof = reader.getTimestampEof(),
        .frame_length = static_cast<unsigned>(reader.getFrameLength()),
        .integ_lines = static_cast<unsigned>(reader.getIntegLines()),
        .global_gain = static_cast<unsigned>(reader.getGlobalGain()),
    };

    auto *buf = &rear_camera->camera_bufs[buf_idx];
    uint8_t *const rgb = static_cast<uint8_t *>(buf->addr);

    // Convert to RGB.
    const int result = libyuv::I420ToRGB24(y, width, u, width / 2, v, width / 2,
                                           rgb, width * 3, width, height);
    assert(result == 0);

    visionbuf_sync(buf, VISIONBUF_SYNC_TO_DEVICE);

    // HACK(mgraczyk): Do not drop frames.
    while (*(volatile int*)&tb->pending_idx != -1) {
      usleep(20000);
    }
    tbuffer_dispatch(tb, buf_idx);
  }

  zsock_destroy(&frame_sock);
}

}  // namespace

CameraInfo cameras_supported[CAMERA_ID_MAX] = {
  [CAMERA_ID_IMX298] = {
      .frame_width = FRAME_WIDTH,
      .frame_height = FRAME_HEIGHT,
      .frame_stride = FRAME_WIDTH*3,
      .bayer = false,
      .bayer_flip = false,
  },
};

void cameras_init(DualCameraState *s) {
  memset(s, 0, sizeof(*s));

  camera_init(&s->rear, CAMERA_ID_IMX298, 20);
  camera_init(&s->front, CAMERA_ID_IMX298, 20);

  if (getenv("SIMULATOR2")) {
    // simulator camera is flipped vertically
    s->rear.transform = (mat3){{
      1.0,  0.0, 0.0,
      0.0, -1.0, s->rear.ci.frame_height - 1.0f,
      0.0,  0.0, 1.0,
    }};
  } else {
    // assume the input is upside-down
    s->rear.transform = (mat3){{
      -1.0,  0.0, s->rear.ci.frame_width - 1.0f,
       0.0, -1.0, s->rear.ci.frame_height - 1.0f,
       0.0,  0.0, 1.0,
    }};
  }
}

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

void camera_autoexposure(CameraState *s, float grey_frac) {}


void cameras_run(DualCameraState *s) {
  set_thread_name("fake_camera");

  if (getenv("SIMULATOR2")) {
    run_simulator(s);
  } else {
    run_unlogger(s);
  }

  cameras_close(s);

}
