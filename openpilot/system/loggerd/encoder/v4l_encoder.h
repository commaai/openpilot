#pragma once

#include <condition_variable>
#include <functional>
#include <mutex>

#include "common/queue.h"
#include "system/loggerd/encoder/encoder.h"

// a VisionIPC buffer is queued at its buffer index, so vb2 keeps it attached and mapped. that's one for each of
// camerad's 18 buffers or the decoder's 16. a source with more shares indices, which only costs a remap
#define BUF_IN_COUNT 18
#define BUF_OUT_COUNT 6

// the Venus encoder. it relies on four AGNOS kernel patches that aren't upstream yet:
// - "media: venus: signal EOS with a real buffer": without it the drain in encoder_close never ends
// - "media: venus: keep capture buffers across encoder drain": without it there are empty buffers after a restart
// - "media: venus: venc: make every I-frame an IDR frame": without it a requested keyframe has no header
// - "media: venus: venc: scale the source to the capture format size": only for the scaled streams (qcamera and the
//   livestreams). without it a smaller encode is a crop, which the constructor asserts against
class V4LEncoder : public VideoEncoder {
public:
  using PacketCallback = std::function<void(uint8_t *, size_t, int64_t, bool, bool)>;
  using InputDoneCallback = std::function<void(VisionBuf *)>;
  struct Options {
    PacketCallback packet_callback;
    InputDoneCallback input_done_callback;
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
  bool is_h265;

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
  // the frame queued at each input index, from its QBUF to its DQBUF
  std::mutex input_lock;
  std::condition_variable input_cv;
  VisionBuf *input_bufs[BUF_IN_COUNT] = {};
};
