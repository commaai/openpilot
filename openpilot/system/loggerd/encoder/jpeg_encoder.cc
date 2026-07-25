#include "system/loggerd/encoder/jpeg_encoder.h"

#include <cassert>
#include <cstring>

#include "common/swaglog.h"

// Lower qscale = higher quality / bigger files for MJPEG.
constexpr int MJPEG_QSCALE = 7;

JpegEncoder::JpegEncoder(const std::string &publish_name, int width, int height)
    : publish_name(publish_name), thumbnail_width(width), thumbnail_height(height) {
  yuv_buffer.resize((thumbnail_width * thumbnail_height * 3) / 2);
  pm = std::make_unique<PubMaster>(std::vector{publish_name.c_str()});

  const AVCodec *codec = avcodec_find_encoder(AV_CODEC_ID_MJPEG);
  assert(codec);

  codec_ctx = avcodec_alloc_context3(codec);
  assert(codec_ctx);
  codec_ctx->width = thumbnail_width;
  codec_ctx->height = thumbnail_height;
  codec_ctx->pix_fmt = AV_PIX_FMT_YUV420P;
  codec_ctx->time_base = (AVRational){1, 1};
  codec_ctx->color_range = AVCOL_RANGE_JPEG;
  codec_ctx->flags |= AV_CODEC_FLAG_QSCALE;
  codec_ctx->global_quality = FF_QP2LAMBDA * MJPEG_QSCALE;

  int err = avcodec_open2(codec_ctx, codec, NULL);
  assert(err >= 0);

  frame = av_frame_alloc();
  assert(frame);
  frame->format = codec_ctx->pix_fmt;
  frame->width = thumbnail_width;
  frame->height = thumbnail_height;
  frame->linesize[0] = thumbnail_width;
  frame->linesize[1] = thumbnail_width / 2;
  frame->linesize[2] = thumbnail_width / 2;
  frame->color_range = AVCOL_RANGE_JPEG;

  pkt = av_packet_alloc();
  assert(pkt);
}

JpegEncoder::~JpegEncoder() {
  av_packet_free(&pkt);
  av_frame_free(&frame);
  avcodec_free_context(&codec_ctx);
}

void JpegEncoder::pushThumbnail(VisionBuf *buf, const VisionIpcBufExtra &extra) {
  generateThumbnail(buf->y, buf->uv, buf->width, buf->height, buf->stride);

  MessageBuilder msg;
  auto thumbnaild = msg.initEvent().initThumbnail();
  thumbnaild.setFrameId(extra.frame_id);
  thumbnaild.setTimestampEof(extra.timestamp_eof);
  thumbnaild.setThumbnail({out_buffer.data(), out_buffer.size()});

  pm->send(publish_name.c_str(), msg);
}

void JpegEncoder::generateThumbnail(const uint8_t *y_addr, const uint8_t *uv_addr, int width, int height, int stride) {
  int downscale = width / thumbnail_width;
  assert(downscale * thumbnail_height == height);

  uint8_t *y_plane = yuv_buffer.data();
  uint8_t *u_plane = y_plane + thumbnail_width * thumbnail_height;
  uint8_t *v_plane = u_plane + (thumbnail_width * thumbnail_height) / 4;
  {
    // subsampled conversion from nv12 to yuv420p
    for (int hy = 0; hy < thumbnail_height / 2; hy++) {
      for (int hx = 0; hx < thumbnail_width / 2; hx++) {
        int ix = hx * downscale + (downscale - 1) / 2;
        int iy = hy * downscale + (downscale - 1) / 2;
        y_plane[(hy * 2 + 0) * thumbnail_width + (hx * 2 + 0)] = y_addr[(iy * 2 + 0) * stride + ix * 2 + 0];
        y_plane[(hy * 2 + 0) * thumbnail_width + (hx * 2 + 1)] = y_addr[(iy * 2 + 0) * stride + ix * 2 + 1];
        y_plane[(hy * 2 + 1) * thumbnail_width + (hx * 2 + 0)] = y_addr[(iy * 2 + 1) * stride + ix * 2 + 0];
        y_plane[(hy * 2 + 1) * thumbnail_width + (hx * 2 + 1)] = y_addr[(iy * 2 + 1) * stride + ix * 2 + 1];
        u_plane[hy * thumbnail_width / 2 + hx] = uv_addr[iy * stride + ix * 2 + 0];
        v_plane[hy * thumbnail_width / 2 + hx] = uv_addr[iy * stride + ix * 2 + 1];
      }
    }
  }

  compressToJpeg(y_plane, u_plane, v_plane);
}

void JpegEncoder::compressToJpeg(uint8_t *y_plane, uint8_t *u_plane, uint8_t *v_plane) {
  frame->data[0] = y_plane;
  frame->data[1] = u_plane;
  frame->data[2] = v_plane;
  // Required for MJPEG qscale to take effect (global_quality alone is not enough).
  frame->quality = FF_QP2LAMBDA * MJPEG_QSCALE;
  frame->pts = AV_NOPTS_VALUE;

  int err = avcodec_send_frame(codec_ctx, frame);
  if (err < 0) {
    LOGE("thumbnail avcodec_send_frame error %d", err);
    out_buffer.clear();
    return;
  }

  av_packet_unref(pkt);
  err = avcodec_receive_packet(codec_ctx, pkt);
  if (err < 0) {
    LOGE("thumbnail avcodec_receive_packet error %d", err);
    out_buffer.clear();
    return;
  }

  out_buffer.assign(pkt->data, pkt->data + pkt->size);
  av_packet_unref(pkt);
}
