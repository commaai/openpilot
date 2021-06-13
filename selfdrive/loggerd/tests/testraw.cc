#include <unistd.h>
#include <zmq.h>

#include <cstdio>
#include <cstdlib>

#include "selfdrive/common/timing.h"
#include "selfdrive/common/util.h"
#include "selfdrive/common/visionipc.h"
#include "selfdrive/loggerd/raw_logger.h"

int main() {
  int err;

  VisionStream stream;

  VisionStreamBufs buf_info;
  while (true) {
    err = visionstream_init(&stream, VISION_STREAM_YUV, false, &buf_info);
    if (err != 0) {
      printf("visionstream fail\n");
      util::sleep_for(100);
    }
    break;
  }

  RawLogger vidlogger("prcamera", buf_info.width, buf_info.height, 20);
  vidlogger.Open("o1");

  for (int cnt=0; cnt<200; cnt++) {
    VisionIpcBufExtra extra;
    VIPSBuf* buf = visionstream_get(&stream, &extra);
    if (buf == NULL) {
      printf("visionstream get failed\n");
      break;
    }

    if (cnt == 100) {
      vidlogger.Rotate("o2", 2);
    }

    uint8_t *y = (uint8_t*)buf->addr;
    uint8_t *u = y + (buf_info.width*buf_info.height);
    uint8_t *v = u + (buf_info.width/2)*(buf_info.height/2);

    double t1 = millis_since_boot();
    vidlogger.LogFrame(cnt, y, u, v, NULL);
    double t2 = millis_since_boot();
    printf("%d %.2f\n", cnt, (t2-t1));
  }

  vidlogger.Close();

  visionstream_destroy(&stream);

  return 0;
}
