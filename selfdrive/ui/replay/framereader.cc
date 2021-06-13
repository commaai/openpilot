#include "selfdrive/ui/replay/framereader.h"

#include <unistd.h>

#include <cassert>

#include <QDebug>

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

static AVInitializer av_initializer;

FrameReader::FrameReader(const std::string &url, QObject *parent) : url_(url), QObject(parent) {
  process_thread_ = QThread::create(&FrameReader::process, this);
  connect(process_thread_, &QThread::finished, process_thread_, &QThread::deleteLater);
  process_thread_->start();
}

FrameReader::~FrameReader() {
  // wait until thread is finished.
  exit_ = true;
  process_thread_->wait();
  cv_decode_.notify_all();
  cv_frame_.notify_all();
  if (decode_thread_.joinable()) {
    decode_thread_.join();
  }

  // free all.
  for (auto &f : frames_) {
    av_free_packet(&f.pkt);
    if (f.data) {
      delete[] f.data;
    }
  }
  while (!buffer_pool.empty()) {
    delete[] buffer_pool.front();
    buffer_pool.pop();
  }
  av_frame_free(&frmRgb_);
  avcodec_close(pCodecCtx_);
  avcodec_free_context(&pCodecCtx_);
  avformat_close_input(&pFormatCtx_);
  sws_freeContext(sws_ctx_);
}

void FrameReader::process() {
  if (processFrames()) {
    decode_thread_ = std::thread(&FrameReader::decodeThread, this);
  }
  if (!exit_) {
    emit finished();
  }
}

bool FrameReader::processFrames() {
  if (avformat_open_input(&pFormatCtx_, url_.c_str(), NULL, NULL) != 0) {
    qDebug() << "error loading " << url_.c_str();
    return false;
  }
  avformat_find_stream_info(pFormatCtx_, NULL);
  av_dump_format(pFormatCtx_, 0, url_.c_str(), 0);

  auto pCodecCtxOrig = pFormatCtx_->streams[0]->codec;
  auto pCodec = avcodec_find_decoder(pCodecCtxOrig->codec_id);
  assert(pCodec);

  pCodecCtx_ = avcodec_alloc_context3(pCodec);
  int ret = avcodec_copy_context(pCodecCtx_, pCodecCtxOrig);
  assert(ret == 0);

  ret = avcodec_open2(pCodecCtx_, pCodec, NULL);
  assert(ret >= 0);

  width = pCodecCtxOrig->width;
  height = pCodecCtxOrig->height;

  sws_ctx_ = sws_getContext(width, height, AV_PIX_FMT_YUV420P,
                            width, height, AV_PIX_FMT_BGR24,
                            SWS_BILINEAR, NULL, NULL, NULL);
  assert(sws_ctx_);

  frmRgb_ = av_frame_alloc();
  assert(frmRgb_);

  frames_.reserve(60 * 20);  // 20fps, one minute
  do {
    Frame &frame = frames_.emplace_back();
    if (av_read_frame(pFormatCtx_, &frame.pkt) < 0) {
      frames_.pop_back();
      break;
    }
  } while (!exit_);

  valid_ = !exit_;
  return valid_;
}

uint8_t *FrameReader::get(int idx) {
  if (!valid_ || idx < 0 || idx >= frames_.size()) {
    return nullptr;
  }

  {
    std::unique_lock lk(mutex_);
    decode_idx_ = idx;
    cv_decode_.notify_one();
    cv_frame_.wait(lk, [=] { return exit_ || frames_[idx].data || frames_[idx].failed; });
  }

  return frames_[idx].data;
}

void FrameReader::decodeThread() {
  int idx = 0;
  while (!exit_) {
    const int from = std::max(idx, 0);
    const int to = std::min(from + 20, (int)frames_.size());
    for (int i = 0; i < frames_.size() && !exit_; ++i) {
      Frame &frame = frames_[i];
      if (i >= from && i < to) {
        if (frame.data || frame.failed) continue;

        uint8_t *dat = decodeFrame(&frame.pkt);
        std::unique_lock lk(mutex_);
        frame.data = dat;
        frame.failed = !dat;
        cv_frame_.notify_all();
      } else if (frame.data) {
        buffer_pool.push(frame.data);
        frame.data = nullptr;
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

uint8_t *FrameReader::decodeFrame(AVPacket *pkt) {
  int gotFrame;
  AVFrame *f = av_frame_alloc();
  avcodec_decode_video2(pCodecCtx_, f, &gotFrame, pkt);

  uint8_t *dat = nullptr;
  if (gotFrame) {
    if (!buffer_pool.empty()) {
      dat = buffer_pool.front();
      buffer_pool.pop();
    } else {
      dat = new uint8_t[getRGBSize()];
    }

    int ret = avpicture_fill((AVPicture *)frmRgb_, dat, AV_PIX_FMT_BGR24, f->width, f->height);
    assert(ret > 0);
    if (sws_scale(sws_ctx_, (const uint8_t **)f->data, f->linesize, 0,
                  f->height, frmRgb_->data, frmRgb_->linesize) <= 0) {
      delete[] dat;
      dat = nullptr;
    }
  }
  av_frame_free(&f);
  return dat;
}
