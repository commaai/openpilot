#pragma once

#include <atomic>
#include <functional>
#include <initializer_list>
#include <memory>
#include <vector>

#include <linux/videodev2.h>

#include "common/queue.h"
#include "system/loggerd/encoder/encoder.h"
#include "system/loggerd/encoder/v4l_util.h"

#define BUF_OUT_COUNT 6

// a V4L2 stateful encoder. the queues, the bitstream handling and publishing are here, and what differs between the
// kernels' drivers is in VenusEncoder (the upstream kernel's Venus driver) and MsmVidcEncoder (the downstream kernel's
// msm_vidc), picked by create(). a subclass's destructor calls encoder_close(), so the drain is over before the
// subclass is gone
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

  // the encoder for this device's kernel
  static std::unique_ptr<V4LEncoder> create(const EncoderInfo &encoder_info, int in_width, int in_height, Options options);
  static std::unique_ptr<V4LEncoder> create(const EncoderInfo &encoder_info, int in_width, int in_height) {
    return create(encoder_info, in_width, in_height, Options());
  }
  ~V4LEncoder() override;
  int encode_frame(VisionBuf* buf, VisionIpcBufExtra *extra) override;
  void encoder_open() override;
  void encoder_close() override;
  void set_bitrate(int bitrate) override;
  void request_keyframe() override;

protected:
  V4LEncoder(const EncoderInfo &encoder_info, int in_width, int in_height, Options options, int fd, v4l2_memory memory,
             int input_count);

  // the setup, for the subclass constructors: the bitstream format (at the output size), the frame rate and the input
  // format (at the input size), then the controls, then start()
  void set_formats(uint32_t input_format);
  void set_controls(std::initializer_list<v4l2_control> ctrls);
  // requests the buffers, starts streaming and queues the bitstream buffers. requery_bitstream_size is for drivers that
  // only know the bitstream buffer size once the buffers are requested (Venus' comes from the firmware)
  void start(bool requery_bitstream_size);

  // what a dequeued bitstream buffer holds
  struct Bitstream {
    size_t header_len = 0;  // the codec header at its start
    bool frame = false;     // a frame after the header
    bool eos = false;       // the end of the stream that encoder_close drains
  };
  virtual Bitstream parse_bitstream(uint32_t flags, const uint8_t *dat, size_t len) = 0;
  // the input index to queue buf at. blocks until it's free
  virtual int acquire_input(const VisionBuf *buf) = 0;
  // points a V4L2 plane at a buffer
  virtual void set_plane(v4l2_plane *plane, const VisionBuf *buf) = 0;
  virtual uint32_t keyframe_control() const = 0;
  // in encoder_open, before the bitstream is dequeued: restarts the encoder after the previous segment's drain
  virtual void resume() {}

  const int fd;
  const v4l2_memory memory;
  bool is_h265;
  v4l2_format fmt_in = {}, fmt_out = {};
  InputSlots in_slots;

private:
  bool is_open = false;
  int segment_num = -1;
  int counter = 0;
  int current_bitrate = -1;
  SafeQueue<VisionIpcBufExtra> extras;
  PacketCallback packet_callback;
  InputDoneCallback input_done_callback;

  void queue_buffer(v4l2_buf_type buf_type, unsigned int index, VisionBuf *buf, struct timeval timestamp = {});
  void request_buffers(v4l2_buf_type buf_type, unsigned int count);
  static void dequeue_handler(V4LEncoder *e);
  std::thread dequeue_handler_thread;

  VisionBuf buf_out[BUF_OUT_COUNT];
  std::vector<std::atomic<VisionBuf *>> input_bufs;
};
