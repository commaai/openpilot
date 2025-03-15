#pragma once

// has to be in this order
#ifdef __linux__
#include "third_party/linux/include/v4l2-controls.h"
#include <linux/videodev2.h>
#else
#define V4L2_BUF_FLAG_KEYFRAME 8
#endif

#include <cassert>
#include <cstdint>
#include <memory>
#include <thread>
#include <vector>

#include "cereal/messaging/messaging.h"
#include "msgq/visionipc/visionipc.h"
#include "common/queue.h"
#include "system/loggerd/loggerd.h"

class VideoEncoder {
public:
  VideoEncoder(const EncoderInfo &encoder_info, int in_width, int in_height);
  virtual ~VideoEncoder() {}
  virtual int encode_frame(VisionBuf* buf, VisionIpcBufExtra *extra) = 0;
  virtual void encoder_open() = 0;
  virtual void encoder_close() = 0;

  void publisher_publish(int segment_num, uint32_t idx, VisionIpcBufExtra &extra, unsigned int flags, kj::ArrayPtr<capnp::byte> header, kj::ArrayPtr<capnp::byte> dat);

protected:
  int in_width, in_height;
  int out_width, out_height;
  const EncoderInfo encoder_info;

private:
  // total frames encoded
  int cnt = 0;
  std::unique_ptr<PubMaster> pm;
  std::vector<capnp::byte> msg_cache;
};
