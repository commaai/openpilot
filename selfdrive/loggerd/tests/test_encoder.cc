#include <assert.h>
#include <unistd.h>
#include <zmq.h>
#include <cstdio>
#include <cstdlib>
#include <sys/stat.h>
#include "common/timing.h"
#include "common/visionipc.h"
#include "encoder.h"
#define MAIN_FPS 20
#define MAIN_BITRATE 5000000
#define QCAM_BITRATE 128000
const int segment_length = 60;
LogCameraInfo cameras_logged[LOG_CAMERA_ID_MAX] = {
    [LOG_CAMERA_ID_FCAMERA] = {
      .stream_type = VISION_STREAM_YUV,
      .filename = "fcamera.hevc",
      .frame_packet_name = "frame",
      .encode_idx_name = "encodeIdx",
      .fps = 20,
      .bitrate = 5000000,
      .is_h265 = true,
      .downscale = false,
      .has_qcamera = true},
  [LOG_CAMERA_ID_QCAMERA] = {
    .filename = "qcamera.ts",
    .fps = MAIN_FPS,
    .bitrate = QCAM_BITRATE,
    .is_h265 = false,
    .downscale = true,
#ifndef QCOM2
    .frame_width = 480, .frame_height = 360
#else
    .frame_width = 526, .frame_height = 330 // keep pixel count the same?
#endif
  },
};

int main() {
  const char *output_path = "./output";
  mkdir(output_path, 0777);

  EncoderState encoder = {};
  EncoderState encoder_alt = {};

  VisionStream stream;
  VisionStreamBufs buf_info;
  while (true) {
    if (visionstream_init(&stream, VISION_STREAM_YUV, false, &buf_info) != 0) {
      printf("visionstream fail\n");
      usleep(100000);
      continue;
    }
    break;
  }

  encoder_init(&encoder, &cameras_logged[LOG_CAMERA_ID_FCAMERA], buf_info.width, buf_info.height);
  encoder_init(&encoder_alt, &cameras_logged[LOG_CAMERA_ID_QCAMERA], buf_info.width, buf_info.height);

  encoder_open(&encoder, output_path);
  encoder_open(&encoder_alt, output_path);
  double t1 = millis_since_boot();
  int cnt = 0;
  for (; cnt < segment_length * MAIN_FPS; cnt++) {
    VIPCBufExtra extra;
    VIPCBuf *buf = visionstream_get(&stream, &extra);
    if (buf == NULL) {
      printf("visionstream get failed\n");
      break;
    }
    uint8_t *y = (uint8_t *)buf->addr;
    uint8_t *u = y + (buf_info.width * buf_info.height);
    uint8_t *v = u + (buf_info.width / 2) * (buf_info.height / 2);
    encoder_encode_frame(&encoder, y, u, v, &extra);
    encoder_encode_frame(&encoder_alt, y, u, v, &extra);
  }
  encoder_close(&encoder);
  encoder_close(&encoder_alt);

  double t2 = millis_since_boot();
  printf("finish. time: %.0f ms, cnt: %d, avg: %.0fms \n", t2 - t1, cnt, (t2 - t1) / cnt);

  visionstream_destroy(&stream);

  encoder_destroy(&encoder);
  encoder_destroy(&encoder_alt);
  return 0;
}
