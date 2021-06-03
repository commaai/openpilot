#include "selfdrive/ui/replay/framereader.h"

#include <assert.h>
#include <unistd.h>

#include <QDebug>

#include "selfdrive/common/timing.h"

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
    if (f.picture) av_frame_free(&f.picture);
  }
  avcodec_free_context(&pCodecCtx_);
  avformat_free_context(pFormatCtx_);
  sws_freeContext(sws_ctx_);
}

void FrameReader::process() {
  bool success = processFrames();
  if (success) {
    decode_thread_ = std::thread(&FrameReader::decodeThread, this);
  }
  // if (!exit_) {
    emit finished(success);
  // }
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
  assert(pCodec != NULL);

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
  assert(sws_ctx_ != NULL);

  frames_.reserve(60 * 20);  // 20fps, one minute
  do {
    Frame &frame = frames_.emplace_back();
    if (av_read_frame(pFormatCtx_, &frame.pkt) < 0) {
      frames_.pop_back();
      break;
    }
  } while (!exit_);

  valid_ = !exit_;
  return !exit_;
}

void FrameReader::decodeThread() {
  int idx = 0;
  while (!exit_) {
    const int from = std::max(idx, 0);
    const int to = std::min(idx + 15, (int)frames_.size());
    for (int i = from; i < to && !exit_; ++i) {
      Frame &frame = frames_[i];
      if (frame.picture != nullptr || frame.failed) continue;

      int gotFrame;
      AVFrame *pFrame = av_frame_alloc();
      avcodec_decode_video2(pCodecCtx_, pFrame, &gotFrame, &frame.pkt);
      av_free_packet(&frame.pkt);

      AVFrame *picture = gotFrame ? toRGB(pFrame) : nullptr;
      av_frame_free(&pFrame);

      if (!picture) {
        qDebug() << "failed to decode frame " << i << " in " << url_.c_str();
      }
      std::unique_lock lk(mutex_);
      frame.picture = picture;
      frame.failed = !picture;
      cv_frame_.notify_all();
    }

    // sleep & wait
    std::unique_lock lk(mutex_);
    cv_decode_.wait(lk, [=] { return exit_ || decode_idx_ != -1; });
    idx = decode_idx_;
    decode_idx_ = -1;
  }
}

AVFrame *FrameReader::toRGB(AVFrame *frm) {
  int numBytes = avpicture_get_size(AV_PIX_FMT_BGR24, frm->width, frm->height);
  if (numBytes > 0) {
    uint8_t *buffer = (uint8_t *)av_malloc(numBytes * sizeof(uint8_t));
    AVFrame *frmRgb = av_frame_alloc();
    if (avpicture_fill((AVPicture *)frmRgb, buffer, AV_PIX_FMT_BGR24, frm->width, frm->height) > 0 &&
        sws_scale(sws_ctx_, (uint8_t const *const *)frm->data, frm->linesize, 0, frm->height,
                  frmRgb->data, frmRgb->linesize) > 0) {
      return frmRgb;
    }
    av_frame_free(&frmRgb);
  }
  return nullptr;
}

uint8_t *FrameReader::get(int idx) {
  if (!valid_ || idx < 0 || idx >= frames_.size()) return nullptr;

  std::unique_lock lk(mutex_);
  decode_idx_ = idx;
  cv_decode_.notify_one();
  double t1 = millis_since_boot();
  cv_frame_.wait(lk, [=] { return exit_ || frames_[idx].picture != nullptr || frames_[idx].failed; });
  if (double dt = millis_since_boot() - t1; dt > 20) {
    qDebug() << "slow get frame " << idx << ", time: " << dt << "ms";
  }

  return frames_[idx].picture ? frames_[idx].picture->data[0] : nullptr;
}
