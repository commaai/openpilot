#pragma once

#include <cassert>
#include <cstdint>
#include <thread>

#include "cereal/messaging/messaging.h"
#include "cereal/visionipc/visionipc.h"
#include "selfdrive/common/queue.h"
#include "selfdrive/loggerd/video_writer.h"
#include "selfdrive/camerad/cameras/camera_common.h"

#define V4L2_BUF_FLAG_KEYFRAME 8

class VideoEncoder {
public:
  virtual ~VideoEncoder() {}
  virtual int encode_frame(const uint8_t *y_ptr, const uint8_t *u_ptr, const uint8_t *v_ptr,
                           int in_width, int in_height, VisionIpcBufExtra *extra) = 0;
  virtual void encoder_open(const char* path) = 0;
  virtual void encoder_close() = 0;

  void publisher_init();
  static void publisher_publish(VideoEncoder *e, int segment_num, uint32_t idx, VisionIpcBufExtra &extra, unsigned int flags, kj::ArrayPtr<capnp::byte> header, kj::ArrayPtr<capnp::byte> dat);

  void writer_open(const char* path) {
    if (this->write) write_handler_thread = std::thread(VideoEncoder::write_handler, this, path);
  }

  void writer_close() {
    if (this->write) {
      to_write.push(NULL);
      write_handler_thread.join();
    }
    assert(to_write.empty());
  }

protected:
  bool write;
  const char* filename;
  int width, height, fps;
  Codec codec;
  CameraType type;

private:
  // publishing
  std::unique_ptr<PubMaster> pm;
  const char *service_name;

  // writing support
  static void write_handler(VideoEncoder *e, const char *path);
  std::thread write_handler_thread;
  SafeQueue<kj::Array<capnp::word>* > to_write;
};
