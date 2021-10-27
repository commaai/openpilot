#include "selfdrive/ui/replay/framereader.h"

#include <unistd.h>
#include <cassert>
#include <mutex>
#include <sstream>

namespace {

int ffmpeg_lockmgr_cb(void **arg, enum AVLockOp op) {
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

int readFunction(void *opaque, uint8_t *buf, int buf_size) {
  auto &iss = *reinterpret_cast<std::istringstream *>(opaque);
  iss.read(reinterpret_cast<char *>(buf), buf_size);
  return iss.gcount() ? iss.gcount() : AVERROR_EOF;
}

} // namespace

FrameReader::FrameReader(bool local_cache, int chunk_size, int retries) : FileReader(local_cache, chunk_size, retries) {
  static std::once_flag once_flag;
  std::call_once(once_flag, [] {
    av_lockmgr_register(ffmpeg_lockmgr_cb);
    av_register_all();
  });

  pFormatCtx_ = avformat_alloc_context();
  av_frame_ = av_frame_alloc();
  rgb_frame_ = av_frame_alloc();
  yuv_frame_ = av_frame_alloc();;
}

FrameReader::~FrameReader() {
  for (auto &f : frames_) {
    av_free_packet(&f.pkt);
  }
  if (pCodecCtx_) avcodec_free_context(&pCodecCtx_);
  if (pFormatCtx_) avformat_close_input(&pFormatCtx_);
  if (av_frame_) av_frame_free(&av_frame_);
  if (rgb_frame_) av_frame_free(&rgb_frame_);
  if (yuv_frame_) av_frame_free(&yuv_frame_);
  if (rgb_sws_ctx_) sws_freeContext(rgb_sws_ctx_);
  if (yuv_sws_ctx_) sws_freeContext(yuv_sws_ctx_);

  if (avio_ctx_) {
    av_freep(&avio_ctx_->buffer);
    av_freep(&avio_ctx_);
  }
}

bool FrameReader::load(const std::string &url, std::atomic<bool> *abort) {
  std::string content = read(url, abort);
  if (content.empty()) return false;

  std::istringstream iss(content);
  const int avio_ctx_buffer_size = 64 * 1024;
  unsigned char *avio_ctx_buffer = (unsigned char *)av_malloc(avio_ctx_buffer_size);
  avio_ctx_ = avio_alloc_context(avio_ctx_buffer, avio_ctx_buffer_size, 0, &iss, readFunction, nullptr, nullptr);
  pFormatCtx_->pb = avio_ctx_;

  pFormatCtx_->probesize = 10 * 1024 * 1024;  // 10MB
  int err = avformat_open_input(&pFormatCtx_, url.c_str(), NULL, NULL);
  if (err != 0) {
    char err_str[1024] = {0};
    av_strerror(err, err_str, std::size(err_str));
    printf("Error loading video - %s - %s\n", err_str, url.c_str());
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

  width = (pCodecCtxOrig->width + 3) & ~3;
  height = pCodecCtxOrig->height;
  rgb_sws_ctx_ = sws_getContext(pCodecCtxOrig->width, pCodecCtxOrig->height, AV_PIX_FMT_YUV420P,
                            width, height, AV_PIX_FMT_BGR24,
                            SWS_FAST_BILINEAR, NULL, NULL, NULL);
  if (!rgb_sws_ctx_) return false;

  yuv_sws_ctx_ = sws_getContext(pCodecCtxOrig->width, pCodecCtxOrig->height, AV_PIX_FMT_YUV420P,
                            width, height, AV_PIX_FMT_YUV420P,
                            SWS_FAST_BILINEAR, NULL, NULL, NULL);
  if (!yuv_sws_ctx_) return false;

  // init buffers
  yuv_buf_.resize(getYUVSize());
  rgb_buf_.resize(getRGBSize());

  frames_.reserve(60 * 20);  // 20fps, one minute
  while (!(abort && *abort)) {
    Frame &frame = frames_.emplace_back();
    err = av_read_frame(pFormatCtx_, &frame.pkt);
    if (err < 0) {
      frames_.pop_back();
      valid_ = (err == AVERROR_EOF);
      break;
    }
    // some stream seems to contian no keyframes
    key_frames_count_ += frame.pkt.flags & AV_PKT_FLAG_KEY;
  }
  return valid_;
}

bool FrameReader::get(int idx, uint8_t *rgb, uint8_t *yuv) {
  assert(rgb != nullptr || yuv != nullptr);
  if (!valid_ || idx < 0 || idx >= frames_.size()) {
    return false;
  }
  return decode(idx, rgb, yuv);
}

void *FrameReader::get(int idx) {
  bool ret = get(idx, rgb_buf_.data(), nullptr);
  return ret ? rgb_buf_.data() : nullptr;
}

bool FrameReader::decode(int idx, uint8_t *rgb, uint8_t *yuv) {
  auto get_keyframe = [=](int idx) {
    for (int i = idx; i >= 0 && key_frames_count_ > 1; --i) {
      if (frames_[i].pkt.flags & AV_PKT_FLAG_KEY) return i;
    }
    return idx;
  };

  int from_idx = idx;
  if (idx > 0 && !frames_[idx].decoded && !frames_[idx - 1].decoded) {
    // find the previous keyframe
    from_idx = get_keyframe(idx);
  }

  for (int i = from_idx; i <= idx; ++i) {
    Frame &frame = frames_[i];
    if ((!frame.decoded || i == idx) && !frame.failed) {
      avcodec_decode_video2(pCodecCtx_, av_frame_, &frame.decoded, &(frame.pkt));
      frame.failed = !frame.decoded;
      if (frame.decoded && i == idx) {
        return decodeFrame(av_frame_, rgb, yuv);
      }
    }
  }
  return false;
}

bool FrameReader::decodeFrame(AVFrame *f, uint8_t *rgb, uint8_t *yuv) {
  // images is going to be written to output buffers, no alignment (align = 1)
  av_image_fill_arrays(yuv_frame_->data, yuv_frame_->linesize, yuv, AV_PIX_FMT_YUV420P, width, height, 1);
  int ret = sws_scale(yuv_sws_ctx_, (const uint8_t **)f->data, f->linesize, 0, f->height, yuv_frame_->data, yuv_frame_->linesize);
  if (ret < 0) return false;

  av_image_fill_arrays(rgb_frame_->data, rgb_frame_->linesize, rgb, AV_PIX_FMT_BGR24, width, height, 1);
  ret = sws_scale(rgb_sws_ctx_, (const uint8_t **)f->data, f->linesize, 0, f->height, rgb_frame_->data, rgb_frame_->linesize);
  return ret >= 0;
}
