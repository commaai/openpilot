#pragma once

#include "common/queue.h"
#include "msgq/visionipc/visionipc.h"
#include "system/loggerd/loggerd.h"

#define BURN_BUF_IN_COUNT 32   // Increased buffer count for parallel processing
#define BURN_BUF_OUT_COUNT 32  // More output buffers for stress testing

// Burn encoder that doesn't set up publishing to avoid service registry issues
class V4LEncoderBurn {
public:
  V4LEncoderBurn(const EncoderInfo &encoder_info, int in_width, int in_height);
  ~V4LEncoderBurn();
  int encode_frame(VisionBuf* buf, VisionIpcBufExtra *extra);
  void encoder_open();
  void encoder_close();

private:
  int fd;
  int out_width, out_height;
  const EncoderInfo encoder_info;

  bool is_open = false;
  int segment_num = -1;
  int counter = 0;

  SafeQueue<VisionIpcBufExtra> extras;

  static void dequeue_handler(V4LEncoderBurn *e);
  std::thread dequeue_handler_thread;

  VisionBuf buf_out[BURN_BUF_OUT_COUNT];
  SafeQueue<unsigned int> free_buf_in;
};