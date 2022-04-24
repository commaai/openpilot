#pragma once

#include "selfdrive/common/queue.h"
#include "selfdrive/loggerd/encoder.h"
#include "selfdrive/loggerd/loggerd.h"
#include "selfdrive/loggerd/video_writer.h"

#define BUF_IN_COUNT 7
#define BUF_OUT_COUNT 6

class V4LEncoder : public VideoEncoder {
public:
  V4LEncoder(const char* filename, CameraType type, int width, int height, int fps, int bitrate, bool h265, int out_width, int out_height, bool write = true);
  ~V4LEncoder();
  int encode_frame(const uint8_t *y_ptr, const uint8_t *u_ptr, const uint8_t *v_ptr,
                   int in_width, int in_height, uint64_t ts);
  void encoder_open(const char* path);
  void encoder_close();
private:
  int fd;

  const char* filename;
  CameraType type;
  unsigned int in_width_, in_height_;
  unsigned int width, height, fps;
  bool remuxing, write;
  bool is_open = false;
  int segment_num = -1;

  std::unique_ptr<PubMaster> pm;
  const char *service_name;

  static void dequeue_handler(V4LEncoder *e);
  std::thread dequeue_handler_thread;

  VisionBuf buf_in[BUF_IN_COUNT];
  VisionBuf buf_out[BUF_OUT_COUNT];

  int buffer_in = 0;
  std::atomic<int> buffer_in_outstanding = 0;

  static void write_handler(V4LEncoder *e, const char *path);
  std::thread write_handler_thread;
  SafeQueue<kj::Array<capnp::word>* > to_write;
};
