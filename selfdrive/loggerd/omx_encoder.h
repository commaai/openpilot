#pragma once

#include <cstdint>
#include <cstdio>
#include <vector>
#include <thread>

#include <OMX_Component.h>

#include "selfdrive/common/queue.h"
#include "selfdrive/loggerd/encoder.h"
#include "selfdrive/loggerd/video_writer.h"

struct OmxBuffer {
  OMX_BUFFERHEADERTYPE header;
  OMX_U8 data[];
};


// OmxEncoder, lossey codec using hardware hevc
class OmxEncoder : public VideoEncoder {
public:
  OmxEncoder(const char* filename, CameraType type, int width, int height, int fps, int bitrate, bool h265, int out_width, int out_height, bool write = true);
  ~OmxEncoder();
  int encode_frame(const uint8_t *y_ptr, const uint8_t *u_ptr, const uint8_t *v_ptr,
                   int in_width, int in_height, uint64_t ts);
  void encoder_open(const char* path);
  void encoder_close();

  // OMX callbacks
  static OMX_ERRORTYPE event_handler(OMX_HANDLETYPE component, OMX_PTR app_data, OMX_EVENTTYPE event,
                                     OMX_U32 data1, OMX_U32 data2, OMX_PTR event_data);
  static OMX_ERRORTYPE empty_buffer_done(OMX_HANDLETYPE component, OMX_PTR app_data,
                                         OMX_BUFFERHEADERTYPE *buffer);
  static OMX_ERRORTYPE fill_buffer_done(OMX_HANDLETYPE component, OMX_PTR app_data,
                                        OMX_BUFFERHEADERTYPE *buffer);

private:
  void wait_for_state(OMX_STATETYPE state);
  static void callback_handler(OmxEncoder *e);
  static void write_and_broadcast_handler(OmxEncoder *e);
  static void handle_out_buf(OmxEncoder *e, OmxBuffer *out_buf);

  int in_width_, in_height_;
  int width, height, fps;
  bool is_open = false;
  bool dirty = false;
  bool write = false;
  int counter = 0;
  std::thread callback_handler_thread;
  std::thread write_handler_thread;
  int segment_num = -1;
  std::unique_ptr<PubMaster> pm;
  const char *service_name;

  const char* filename;
  CameraType type;

  std::mutex state_lock;
  std::condition_variable state_cv;
  OMX_STATETYPE state = OMX_StateLoaded;

  OMX_HANDLETYPE handle;

  std::vector<OMX_BUFFERHEADERTYPE *> in_buf_headers;
  std::vector<OMX_BUFFERHEADERTYPE *> out_buf_headers;

  uint64_t last_t;

  SafeQueue<OMX_BUFFERHEADERTYPE *> free_in;
  SafeQueue<OMX_BUFFERHEADERTYPE *> done_out;
  SafeQueue<OmxBuffer *> to_write;

  bool remuxing;
  std::unique_ptr<VideoWriter> writer;

  bool downscale;
  uint8_t *y_ptr2, *u_ptr2, *v_ptr2;
};
