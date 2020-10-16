#include <assert.h>
#include <unistd.h>
#include <zmq.h>

#include <cstdio>
#include <cstdlib>

#include "common/timing.h"
#include "common/visionipc.h"
#include "encoder.h"

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
};

int main() {
  int err;

  EncoderState state;
  EncoderState *s = &state;
  memset(s, 0, sizeof(*s));

  int w = 1164;
  int h = 874;
  FILE *infile = fopen("/sdcard/camera_t2.yuv", "rb");
  assert(infile);
  uint8_t *inbuf = (uint8_t *)malloc(w * h * 3 / 2);
  encoder_init(s, &cameras_logged[LOG_CAMERA_ID_FCAMERA], w, h);
  double t1 = millis_since_boot();
  encoder_open(s, "/data/openpilot/selfdrive/loggerd/tests");
  int cnt = 0;
  uint64_t timestamp = nanos_since_boot();
  while (true) {
    ssize_t size = fread(inbuf, w * h * 3 / 2, 1, infile);
    if (size != 1) {
     
      break;
    }
    uint8_t *tmpy = inbuf;
    uint8_t *tmpu = inbuf + w * h;
    uint8_t *tmpv = inbuf + w * h + (w / 2) * (h / 2);
    VIPCBufExtra extra;
    extra.timestamp_eof = timestamp;
    encoder_encode_frame(s, tmpy, tmpu, tmpv, &extra);
    timestamp += 50 *1000;
    ++cnt;
  }
  encoder_close(s);
  double t2 = millis_since_boot();
  printf("finish. time: %f ms, cnt: %d, avg: %f \n", t2-t1, cnt, (t2-t1)/cnt);
  encoder_destroy(s);
  fclose(infile);
  free(inbuf);
  return 0;
}
