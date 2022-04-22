#pragma once

#include "selfdrive/loggerd/encoder.h"
#include "selfdrive/loggerd/loggerd.h"
#include "selfdrive/loggerd/video_writer.h"

// has to be in this order
#include "selfdrive/loggerd/include/v4l2-controls.h"
#include <linux/videodev2.h>

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
  const char* filename;
  CameraType type;
  unsigned int in_width_, in_height_;
  unsigned int width, height, fps;
  bool remuxing, write;
  bool is_open = false;

  std::unique_ptr<VideoWriter> writer;
  int fd;

  std::unique_ptr<PubMaster> pm;
  const char *service_name;

  static void dequeue_out_handler(V4LEncoder *e);
  std::thread dequeue_out_thread;

  static void dequeue_in_handler(V4LEncoder *e);
  std::thread dequeue_in_thread;

  int queue_buffer(v4l2_buf_type buf_type, unsigned int index, VisionBuf *buf, unsigned int bytesused=0, struct timeval timestamp={0});
  int dequeue_buffer(v4l2_buf_type buf_type, unsigned int *index=NULL, unsigned int *bytesused=NULL, unsigned int *flags=NULL, struct timeval *timestamp=NULL);

  VisionBuf buf_in[BUF_IN_COUNT];
  VisionBuf buf_out[BUF_OUT_COUNT];

  int buffer_in = 0;
};
