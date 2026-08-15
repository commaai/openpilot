#pragma once

#include <atomic>
#include <functional>

#include "common/queue.h"
#include "system/loggerd/encoder/encoder.h"

#define BUF_IN_COUNT 9
#define BUF_OUT_COUNT 6

class V4LEncoder : public VideoEncoder {
public:
  using PacketCallback = std::function<void(uint8_t *, size_t, int64_t, bool, bool)>;
  using InputDoneCallback = std::function<void(VisionBuf *)>;
  struct Options {
    PacketCallback packet_callback;
    uint32_t input_format = V4L2_PIX_FMT_NV12;
    InputDoneCallback input_done_callback;
    bool max_performance = false;
  };

  V4LEncoder(const EncoderInfo &encoder_info, int in_width, int in_height);
  V4LEncoder(const EncoderInfo &encoder_info, int in_width, int in_height, Options options);
  ~V4LEncoder();
  int encode_frame(VisionBuf* buf, VisionIpcBufExtra *extra);
  void encoder_open();
  void encoder_close();
  void set_bitrate(int bitrate);
  void request_keyframe();

private:
  int fd;

  bool is_open = false;
  int segment_num = -1;
  int counter = 0;
  int current_bitrate = -1;
  SafeQueue<VisionIpcBufExtra> extras;
  PacketCallback packet_callback;
  InputDoneCallback input_done_callback;

  static void dequeue_handler(V4LEncoder *e);
  std::thread dequeue_handler_thread;

  VisionBuf buf_out[BUF_OUT_COUNT];
  std::atomic<VisionBuf *> input_bufs[BUF_IN_COUNT] = {};
  SafeQueue<unsigned int> free_buf_in;
};
