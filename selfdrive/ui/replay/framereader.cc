#include "selfdrive/ui/replay/framereader.h"

#include <unistd.h>

#include <cassert>

#include "libyuv.h"

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

struct AVInitializer {
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
  if (decode_thread_.joinable()) {
    decode_thread_.join();
  }

  // free all.
  for (auto &f : frames_) {
    av_free_packet(&f.pkt);
    delete f.buf;
  }
  for (Buffer *buf : buffers_) {
    delete buf;
  }
  if (pCodecCtx_) {
    avcodec_close(pCodecCtx_);
    avcodec_free_context(&pCodecCtx_);
  }
  if (pFormatCtx_) {
    avformat_close_input(&pFormatCtx_);
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

  // pCodecCtx_->thread_count = 0;
  // pCodecCtx_->thread_type = FF_THREAD_FRAME;
  ret = avcodec_open2(pCodecCtx_, pCodec, NULL);
  if (ret < 0) return false;

  width = pCodecCtxOrig->width;
  height = pCodecCtxOrig->height;

  frames_.reserve(60 * 20);  // 20fps, one minute
  do {
    Frame &frame = frames_.emplace_back();
    int err = av_read_frame(pFormatCtx_, &frame.pkt);
    if (err < 0) {
      frames_.pop_back();
      valid_ = (err == AVERROR_EOF);
      break;
    }
    // some stream seems to contian no keyframes
    key_frames_count_ += frame.pkt.flags & AV_PKT_FLAG_KEY;
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
  cv_decode_.wait(lk, [=] { return exit_ || frames_[idx].buf || frames_[idx].failed; });
  if (!frames_[idx].buf) {
    return std::nullopt;
  }
  return std::make_pair(frames_[idx].buf->rgb.get(), frames_[idx].buf->yuv.get());
}

void FrameReader::decodeThread() {
  // find the previous keyframe
  auto get_keyframe = [=](int idx) {
    for (int i = idx; i >= 0 && key_frames_count_ > 1; --i) {
      if (frames_[i].pkt.flags & AV_PKT_FLAG_KEY) return i;
    }
    return idx;
  };

  int frame_id = 0;
  AVFrame *av_frame = av_frame_alloc();
  while (!exit_) {
    {
      std::unique_lock lk(mutex_);
      cv_decode_.wait(lk, [=] { return exit_ || decode_idx_ != frame_id; });
    }

    while (std::exchange(frame_id, decode_idx_) != frame_id) {
      const int from_frame = get_keyframe(frame_id);
      printf("key %d , current %d\n", from_frame, frame_id);
      const int to_frame = std::min(frame_id + 10, (int)frames_.size() - 1);

      for (int i = 0; i < frames_.size() && !exit_ && decode_idx_ == frame_id; ++i) {
        Frame &frame = frames_[i];
        if (i >= from_frame && i <= to_frame) {
          if (!frame.decoded || (i >= frame_id && !frame.buf)) {
            while (true) {
              int ret = avcodec_decode_video2(pCodecCtx_,av_frame, &frame.decoded, &(frame.pkt));
              if (ret > 0 && !frame.decoded) {
                // decode thread is still receiving the initial packets
                usleep(0);
              } else {
                break;
              }
            }
            
            if (i >= frame_id) {
              Buffer *buf = frame.decoded ? decodeFrame(av_frame) : nullptr;
              std::unique_lock lk(mutex_);
              frame.buf = buf;
              frame.failed = !buf;
              cv_decode_.notify_all();
            }
          }
        } else if (frame.buf) {
          buffers_.push_back(frame.buf);
          frame.buf = nullptr;
        }
      }
    }
  }
  av_frame_free(&av_frame);
}

FrameReader::Buffer *FrameReader::decodeFrame(AVFrame *f) {
  Buffer *buf = nullptr;
  if (!buffers_.empty()) {
    buf = buffers_.back();
    buffers_.pop_back();
  } else {
    buf = new Buffer(getRGBSize(), getYUVSize());
  }
  int i, j, k;
  uint8_t *y = buf->yuv.get();
  for (i = 0; i < f->height; i++) {
    memcpy(y + f->width * i, f->data[0] + f->linesize[0] * i, f->width);
  }
  for (j = 0; j < f->height / 2; j++) {
    memcpy(y + f->width * i + f->width / 2 * j, f->data[1] + f->linesize[1] * j, f->width / 2);
  }
  for (k = 0; k < f->height / 2; k++) {
    memcpy(y + f->width * i + f->width / 2 * j + f->width / 2 * k, f->data[2] + f->linesize[2] * k, f->width / 2);
  }

  uint8_t *u = y + f->width * f->height;
  uint8_t *v = u + (f->width / 2) * (f->height / 2);
  libyuv::I420ToRGB24(y, f->width, u, f->width / 2, v, f->width / 2, buf->rgb.get(), f->width * 3, f->width, f->height);
  return buf;
}
