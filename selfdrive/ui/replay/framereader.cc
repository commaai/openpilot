#include "selfdrive/ui/replay/framereader.h"

#include <cassert>
 #include <unistd.h>

struct AVInitializer {
  AVInitializer() {
    av_register_all();
    avformat_network_init();
  }
  ~AVInitializer() { avformat_network_deinit(); }
};

FrameReader::FrameReader() {
  static AVInitializer av_initializer;
}

FrameReader::~FrameReader() {
  for (auto &pkt : packets_) {
    av_free_packet(&pkt);
  }
  if (frmRgb_) {
    av_frame_free(&frmRgb_);
  }
  if (pCodecCtx_) {
    avcodec_close(pCodecCtx_);
    avcodec_free_context(&pCodecCtx_);
  }
  if (pFormatCtx_) {
    avformat_close_input(&pFormatCtx_);
  }
  if (sws_ctx_) {
    sws_freeContext(sws_ctx_);
  }
}

bool FrameReader::load(const std::string &url) {
  pFormatCtx_ = avformat_alloc_context();
  if (avformat_open_input(&pFormatCtx_, url.c_str(), NULL, NULL) != 0) {
    printf("error loading %s\n", url.c_str());
    return false;
  }
  avformat_find_stream_info(pFormatCtx_, NULL);
  // av_dump_format(pFormatCtx_, 0, url.c_str(), 0);

  auto pCodecCtxOrig = pFormatCtx_->streams[0]->codec;
  auto pCodec = avcodec_find_decoder(pCodecCtxOrig->codec_id);
  if (!pCodec) return false;

  pCodecCtx_ = avcodec_alloc_context3(pCodec);

  int ret = avcodec_copy_context(pCodecCtx_, pCodecCtxOrig);
  if (ret != 0) return false;

  pCodecCtx_->thread_count = 0;
  pCodecCtx_->thread_type = FF_THREAD_FRAME;

  ret = avcodec_open2(pCodecCtx_, pCodec, NULL);
  if (ret < 0) return false;

  width = pCodecCtxOrig->width;
  height = pCodecCtxOrig->height;

  sws_ctx_ = sws_getContext(width, height, AV_PIX_FMT_YUV420P,
                            width, height, AV_PIX_FMT_BGR24,
                            SWS_BILINEAR, NULL, NULL, NULL);
  if (!sws_ctx_) return false;

  frmRgb_ = av_frame_alloc();
  if (!frmRgb_) return false;

  packets_.reserve(60 * 20);  // 20fps, one minute
  while (true) {
    AVPacket &pkt = packets_.emplace_back();
    int err = av_read_frame(pFormatCtx_, &pkt);
    if (err < 0) {
      packets_.pop_back();
      valid_ = (err == AVERROR_EOF);
      break;
    }
  };
  return valid_;
}

bool FrameReader::get(int idx, uint8_t *rgb_dat, uint8_t *yuv_dat) {
  if (!valid_ || idx < 0 || idx >= packets_.size()) return false;
  return decodeFrame(&packets_[idx], rgb_dat, yuv_dat);
}

bool FrameReader::decodeFrame(AVPacket *pkt, uint8_t *rgb_dat, uint8_t *yuv_dat) {
  bool success = false;
  int gotFrame = 0;
  AVFrame *f = av_frame_alloc();
  while (true) {
    int ret = avcodec_decode_video2(pCodecCtx_, f, &gotFrame, pkt);
    if (ret > 0 && !gotFrame) {
      // decode thread is still receiving the initial packets
      usleep(0);
    } else {
      break;
    }
  }
  if (gotFrame) {
    if (yuv_dat) {
      int i, j, k;
      for (i = 0; i < f->height; i++) {
        memcpy(yuv_dat + f->width * i, f->data[0] + f->linesize[0] * i, f->width);
      }
      for (j = 0; j < f->height / 2; j++) {
        memcpy(yuv_dat + f->width * i + f->width / 2 * j, f->data[1] + f->linesize[1] * j, f->width / 2);
      }
      for (k = 0; k < f->height / 2; k++) {
        memcpy(yuv_dat + f->width * i + f->width / 2 * j + f->width / 2 * k, f->data[2] + f->linesize[2] * k, f->width / 2);
      }
      success = true;
    }
    if (rgb_dat) {
      int ret = avpicture_fill((AVPicture *)frmRgb_, rgb_dat, AV_PIX_FMT_BGR24, f->width, f->height);
      assert(ret > 0);
      success = sws_scale(sws_ctx_, (const uint8_t **)f->data, f->linesize, 0, f->height, frmRgb_->data, frmRgb_->linesize) > 0;
    }
  }
  av_frame_free(&f);
  return success;
}
