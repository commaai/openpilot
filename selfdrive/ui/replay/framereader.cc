#include "selfdrive/ui/replay/framereader.h"

#include <cassert>
#include <unistd.h>

static int ffmpeg_lockmgr_cb(void **arg, enum AVLockOp op) {
  std::mutex *mutex = (std::mutex *)*arg;
  switch (op) {
  case AV_LOCK_CREATE:
    mutex = new std::mutex();
    break;
  case AV_LOCK_OBTAIN:
    mutex->lock();
    break;
  case AV_LOCK_RELEASE:
    mutex->unlock();
  case AV_LOCK_DESTROY:
    delete mutex;
    break;
  }
  return 0;
}

class AVInitializer {
public:
  AVInitializer() {
    int ret = av_lockmgr_register(ffmpeg_lockmgr_cb);
    assert(ret >= 0);
    av_register_all();
    avformat_network_init();
  }

  ~AVInitializer() { avformat_network_deinit(); }
};

FrameReader::FrameReader() {
  static AVInitializer av_initializer;
}

FrameReader::~FrameReader() {
  // wait until thread is finished.
  exit_ = true;
  cv_decode_.notify_all();
  cv_frame_.notify_all();
  if (decode_thread_.joinable()) {
    decode_thread_.join();
  }

  // free all.
  for (auto &f : frames_) {
    av_free_packet(&f.pkt);
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
  pFormatCtx_->probesize = 10 * 1024 * 1024;  // 10MB
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

  frames_.reserve(60 * 20);  // 20fps, one minute
  do {
    Frame &frame = frames_.emplace_back();
    int err = av_read_frame(pFormatCtx_, &frame.pkt);
    if (err < 0) {
      frames_.pop_back();
      valid_ = (err == AVERROR_EOF);
      break;
    }
  } while (!exit_);

  if (valid_) {
    decode_thread_ = std::thread(&FrameReader::decodeThread, this);
  }
  return valid_;
}

std::optional<std::pair<uint8_t *, uint8_t *>> FrameReader::get(int idx) {
  if (!valid_ || idx < 0 || idx >= frames_.size()) {
    return std::nullopt;
  }
  std::unique_lock lk(mutex_);
  decode_idx_ = idx;
  cv_decode_.notify_one();
  cv_frame_.wait(lk, [=] { return exit_ || frames_[idx].rgb_data || frames_[idx].failed; });
  if (!frames_[idx].rgb_data) {
    return std::nullopt;
  }
  return std::make_pair(frames_[idx].rgb_data.get(), frames_[idx].yuv_data.get());
}

void FrameReader::decodeThread() {
  int idx = 0;
  while (!exit_) {
    // find the previous key frame
    int key_frame = idx;
    for (int i = idx; i >=0; --i) {
      if (frames_[i].pkt.flags & AV_PKT_FLAG_KEY) {
        key_frame = i;
        break;
      }
    }
    const int to = std::min(idx + 20, (int)frames_.size());
    for (int i = 0; i < frames_.size() && !exit_; ++i) {
      Frame &frame = frames_[i];
      if (i >= key_frame && i < to) {
        if (frame.rgb_data || frame.failed) continue;

        auto [rgb_data, yuv_data] = decodeFrame(&frame.pkt);
        std::unique_lock lk(mutex_);
        frame.rgb_data.reset(rgb_data);
        frame.yuv_data.reset(yuv_data);
        frame.failed = !rgb_data;
        cv_frame_.notify_all();
      } else {
        frame.rgb_data.reset(nullptr);
        frame.yuv_data.reset(nullptr);
        frame.failed = false;
      }
    }

    // sleep & wait
    std::unique_lock lk(mutex_);
    cv_decode_.wait(lk, [=] { return exit_ || decode_idx_ != -1; });
    idx = decode_idx_;
    decode_idx_ = -1;
  }
}

std::pair<uint8_t *, uint8_t *> FrameReader::decodeFrame(AVPacket *pkt) {
  uint8_t *rgb_data = nullptr, *yuv_data = nullptr;
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
    rgb_data = new uint8_t[getRGBSize()];
    yuv_data = new uint8_t[getYUVSize()];
    int i, j, k;
    for (i = 0; i < f->height; i++) {
      memcpy(yuv_data + f->width * i, f->data[0] + f->linesize[0] * i, f->width);
    }
    for (j = 0; j < f->height / 2; j++) {
      memcpy(yuv_data + f->width * i + f->width / 2 * j, f->data[1] + f->linesize[1] * j, f->width / 2);
    }
    for (k = 0; k < f->height / 2; k++) {
      memcpy(yuv_data + f->width * i + f->width / 2 * j + f->width / 2 * k, f->data[2] + f->linesize[2] * k, f->width / 2);
    }

    int ret = avpicture_fill((AVPicture *)frmRgb_, rgb_data, AV_PIX_FMT_BGR24, f->width, f->height);
    assert(ret > 0);
    if (sws_scale(sws_ctx_, (const uint8_t **)f->data, f->linesize, 0, f->height, frmRgb_->data, frmRgb_->linesize) <= 0) {
      delete[] rgb_data;
      delete[] yuv_data;
      rgb_data = yuv_data = nullptr;
    }
  }
  av_frame_free(&f);
  return {rgb_data, yuv_data};
}
