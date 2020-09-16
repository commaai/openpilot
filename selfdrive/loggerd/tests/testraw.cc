#include <cstdio>
#include <cstdlib>
#include <unistd.h>

#include <zmq.h>

#include "common/visionipc.h"
#include "common/timing.h"

#include "RawLogger.h"

int main() {
  int err;

  VisionStream stream;

  VisionStreamBufs buf_info;
  RawLogger vidlogger("prcamera", buf_info.width, buf_info.height, 20);
  vidlogger.Open("o1");
  
  for (int cnt=0; cnt<200; cnt++) {
    VIPSBuf* buf = stream.acquire(VISION_STREAM_YUV, false, nullptr);
    if (buf == NULL) {
      break;
    }

    if (cnt == 100) {
      vidlogger.Rotate("o2", 2);
    }

    uint8_t *y = (uint8_t*)buf->addr;
    uint8_t *u = y + (stream.bufs_info.width*stream.bufs_info.height);
    uint8_t *v = u + (stream.bufs_info.width/2)*(stream.bufs_info.height/2);

    double t1 = millis_since_boot();
    vidlogger.LogFrame(cnt, y, u, v, NULL);
    double t2 = millis_since_boot();
    printf("%d %.2f\n", cnt, (t2-t1));
  }

  vidlogger.Close();
  return 0;
}
