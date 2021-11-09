#include <sys/stat.h>

#include <climits>
#include <condition_variable>
#include <sstream>
#include <thread>

#include "catch2/catch.hpp"
#include "cereal/visionipc/visionipc_client.h"
#include "selfdrive/common/timing.h"
#include "selfdrive/common/util.h"
#include "selfdrive/loggerd/encoder.h"
#include "selfdrive/loggerd/raw_logger.h"

TEST_CASE("RawEncoder") {
  VisionIpcClient vipc_client = VisionIpcClient("camerad", VISION_STREAM_YUV_BACK, true);
  while (!vipc_client.connect(false)) {
    util::sleep_for(100);
  }

  const int width = vipc_client.buffers[0].width;
  const int height = vipc_client.buffers[0].height;
  RawLogger logger("video", width, height, 20, 5000000, true, false, true);
  logger.encoder_open("/tmp");
  int i = 0;
  while (true) {
    VisionIpcBufExtra extra = {};
    VisionBuf *buf = vipc_client.recv(&extra);
    if (buf == nullptr) continue;
    logger.encode_frame(buf->y, buf->u, buf->v, buf->width, buf->height, extra.timestamp_eof);
    if (++i == 1200) {
      break;
    }
  }
  logger.encoder_close();
}

class TestRemuxing : public RawLogger {
public:
  TestRemuxing(const char* filename, int width, int height, int fps,
            int bitrate, bool h265, bool downscale, bool write = true) : RawLogger(filename, width, height, fps, bitrate, h265, downscale, write) {}

protected:
  std::unique_ptr<FFmpegEncoder> remuxing;
};

TEST_CASE("Remuxing") {

}
