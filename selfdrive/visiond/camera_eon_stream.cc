#include "camera_eon_stream.h"

#include <string>
#include <unistd.h>
#include <vector>
#include <string.h>

#include <czmq.h>
#include <libyuv.h>
#include <capnp/serialize.h>
#include "cereal/gen/cpp/log.capnp.h"

#include "common/util.h"
#include "common/timing.h"
#include "common/swaglog.h"
#include "buffering.h"

extern "C" {
#include <libavcodec/avcodec.h>
}

extern volatile int do_exit;

#define FRAME_WIDTH 1164
#define FRAME_HEIGHT 874

namespace {
void camera_open(CameraState *s, cl_mem *yuv_cls, bool rear, cl_device_id device_id, cl_context context, cl_command_queue q) {
  assert(yuv_cls);
  s->yuv_cls = yuv_cls;
  s->device_id = device_id;
  s->context = context;
  s->q = q;
}

void camera_close(CameraState *s) {
  tbuffer_stop(&s->camera_tb);
}

void camera_release_buffer(void *cookie, int buf_idx) {
  CameraState *s = static_cast<CameraState *>(cookie);
}

void camera_init(CameraState *s, int camera_id, unsigned int fps) {
  assert(camera_id < ARRAYSIZE(cameras_supported));
  s->ci = cameras_supported[camera_id];
  assert(s->ci.frame_width != 0);

  s->frame_size = s->ci.frame_height * s->ci.frame_stride;
  s->fps = fps;

  tbuffer_init2(&s->camera_tb, FRAME_BUF_COUNT, "frame", camera_release_buffer,
                s);
}

void run_eon_stream(DualCameraState *s) {
  int err;
  uint8_t stream_start[74] =  
{0x00,0x00,0x00,0x01,0x40,0x01,0x0c,0x01,0xff,0xff,0x01,0x60,0x00,0x00,0x03,0x00,0xb0,0x00,0x00,0x03,0x00,0x00,0x03,0x00,0x5d,0xac,0x59,0x00,0x00,0x00,0x01,0x42,0x01,0x01,0x01,0x60,0x00,0x00,0x03,0x00,0xb0,0x00,0x00,0x03,0x00,0x00,0x03,0x00,0x5d,0xa0,0x02,0x50,0x80,0x38,0x1c,0x5c,0x66,0x5a,0xee,0x4c,0x92,0xec,0x80,0x00,0x00,0x00,0x01,0x44,0x01,0xc0,0xf1,0x80,0x04,0x20};

  avcodec_register_all();
  const AVCodec *codec = avcodec_find_decoder_by_name("hevc");
  if(!codec) {
    LOG("hevc decoder not found\n");
    return;
  }
  AVCodecParserContext *parser = av_parser_init(codec->id);
  if (!parser) {
    LOG("parser not found\n");
    return;
  }
  AVCodecContext *dec_ctx = avcodec_alloc_context3(codec);
  assert(dec_ctx);
  err = avcodec_open2(dec_ctx, codec, NULL);
  assert(err >= 0);
  AVFrame *frame = av_frame_alloc();
  assert(frame);

  std::string zmq_uri(">tcp://");
  const char *eon_ip = getenv("EON_IP");
  if(eon_ip)
    zmq_uri += eon_ip;
  else
    zmq_uri += "192.168.1.105";
  zmq_uri += ":9002";
  LOG("Connecting to Eon stream: %s", zmq_uri.c_str());
  zsock_t *frame_sock = zsock_new_sub(zmq_uri.c_str(), "");
  assert(frame_sock);
  void *frame_sock_raw = zsock_resolve(frame_sock);

  CameraState *const rear_camera = &s->rear;

  auto *tb = &rear_camera->camera_tb;
  AVPacket avpkt;
  av_init_packet(&avpkt);
  // send stream start bytes
  avpkt.size = sizeof(stream_start);
  avpkt.data = &stream_start[0];
  while (avpkt.size > 0) {
    int got_frame = 0;
    int len = avcodec_decode_video2(dec_ctx, frame, &got_frame, &avpkt);
    if (len < 0) {
      LOGD("Error while decoding frame\n");
      return;
    }
    avpkt.size -= len;
    avpkt.data += len;
  }
  while (!do_exit) {
    zmq_msg_t t_msg;
    err = zmq_msg_init(&t_msg);
    assert(err == 0);

    zmq_msg_t frame_msg;
    err = zmq_msg_init(&frame_msg);
    assert(err == 0);

    err = zmq_msg_recv(&t_msg, frame_sock_raw, 0);
    if(err == -1)
	    break;
    err = zmq_msg_recv(&frame_msg, frame_sock_raw, 0);
    if(err == -1)
	    break;

    av_init_packet(&avpkt);
    avpkt.size = zmq_msg_size(&frame_msg);
    if (avpkt.size == 0) {
      LOGD("Empty frame msg recved.\n");
      continue;
    }
    avpkt.data = (uint8_t *)zmq_msg_data(&frame_msg);;
    while (avpkt.size > 0) {
      int got_frame = 0;
      int len = avcodec_decode_video2(dec_ctx, frame, &got_frame, &avpkt);
      assert(len >= 0);
      if (got_frame) {
        assert(frame->width == FRAME_WIDTH);
        assert(frame->height == FRAME_HEIGHT);
        const int buf_idx = tbuffer_select(tb);
        rear_camera->camera_bufs_metadata[buf_idx] = {
          .frame_id = (uint32_t)dec_ctx->frame_number,
          .timestamp_eof = nanos_since_boot(),
          .frame_length = 0,
          .integ_lines = 0,
          .global_gain = 0,
        };
        cl_mem yuv_cl = rear_camera->yuv_cls[buf_idx];
        cl_event map_event;
        void *yuv_buf = (void *)clEnqueueMapBuffer(rear_camera->q, yuv_cl, CL_TRUE,
                                                    CL_MAP_WRITE, 0, frame->width * frame->height * 3 / 2,
                                                    0, NULL, &map_event, &err);
        assert(err == 0);
        clWaitForEvents(1, &map_event);
        clReleaseEvent(map_event);
        uint8_t *write_ptr = (uint8_t *)yuv_buf;
        for(int line_idx = 0; line_idx < frame->height; line_idx++) {
          memcpy(write_ptr, frame->data[0] + line_idx * frame->linesize[0], frame->width);
          write_ptr += frame->width;
        }
        for(int line_idx = 0; line_idx < frame->height / 2; line_idx++) {
          memcpy(write_ptr, frame->data[1] + line_idx * frame->linesize[1], frame->width / 2);
          write_ptr += frame->width / 2;
        }
        for(int line_idx = 0; line_idx < frame->height / 2;line_idx++) {
          memcpy(write_ptr, frame->data[2] + line_idx * frame->linesize[2], frame->width / 2);
          write_ptr += frame->width / 2;
        }
        clEnqueueUnmapMemObject(rear_camera->q, yuv_cl, yuv_buf, 0, NULL, &map_event);
        clWaitForEvents(1, &map_event);
        clReleaseEvent(map_event);
        tbuffer_dispatch(tb, buf_idx);

      }
      avpkt.size -= len;
      avpkt.data += len;
    }
    err = zmq_msg_close(&frame_msg);
    assert(err == 0);
    err = zmq_msg_close(&t_msg);
    assert(err == 0);
  }
  zsock_destroy(&frame_sock);
  av_parser_close(parser);
  avcodec_free_context(&dec_ctx);
  av_frame_free(&frame);
}

}  // namespace

CameraInfo cameras_supported[CAMERA_ID_MAX] = {
  [CAMERA_ID_IMX298] = {
      .frame_width = FRAME_WIDTH,
      .frame_height = FRAME_HEIGHT,
      .frame_stride = FRAME_WIDTH*3,
      .bayer = false,
      .bayer_flip = false,
  },
};

void cameras_init(DualCameraState *s) {
  memset(s, 0, sizeof(*s));

  camera_init(&s->rear, CAMERA_ID_IMX298, 20);
  s->rear.transform = (mat3){{
    1.0,  0.0, 0.0,
    0.0, 1.0, 0.0,
    0.0,  0.0, 1.0,
  }};
}

void camera_autoexposure(CameraState *s, float grey_frac) {}

void cameras_open(DualCameraState *s, cl_mem *yuv_cls_rear, cl_device_id device_id, cl_context context, cl_command_queue q) {
  assert(yuv_cls_rear);
  int err;

  camera_open(&s->rear, yuv_cls_rear, true, device_id, context, q);
}

void cameras_close(DualCameraState *s) {
  camera_close(&s->rear);
}

void cameras_run(DualCameraState *s) {
  set_thread_name("Eon streaming");
  run_eon_stream(s);
  cameras_close(s);
}
