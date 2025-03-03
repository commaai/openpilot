#pragma once

#include "common/queue.h"
#include "system/loggerd/encoder/encoder.h"

#define BUF_IN_COUNT 7
#define BUF_OUT_COUNT 6

class V4LEncoder : public VideoEncoder {
public:
  V4LEncoder(const EncoderInfo &encoder_info, int in_width, int in_height);
  ~V4LEncoder();
  int encode_frame(VisionBuf* buf, VisionIpcBufExtra *extra);
  void encoder_open();
  void encoder_close();

private:
  int fd;

  bool is_open = false;
  int segment_num = -1;
  int counter = 0;

  SafeQueue<VisionIpcBufExtra> extras;

  static void dequeue_handler(V4LEncoder *e);
  std::thread dequeue_handler_thread;

  VisionBuf buf_out[BUF_OUT_COUNT];
  SafeQueue<unsigned int> free_buf_in;
};
