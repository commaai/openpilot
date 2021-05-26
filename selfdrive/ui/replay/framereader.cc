#include "selfdrive/ui/replay/framereader.h"

#include <assert.h>
#include <unistd.h>

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

FrameReader::FrameReader(const std::string &url, VisionStreamType stream_type, QObject *parent)
    : url(url), stream_type(stream_type), QThread(parent) {}

FrameReader::~FrameReader() {
  // wait until thread is finished.
  exit_ = true;
  cv_decode.notify_one();
  wait();

  // free all.
  for (auto &f : frames) {
    if (f.picture) av_frame_free(&f.picture);
  }
  avcodec_free_context(&pCodecCtx);
  avformat_free_context(pFormatCtx);
  sws_freeContext(sws_ctx);
}

void FrameReader::run() {
  processFrames();
  decodeFrames();
}

void FrameReader::processFrames() {
  if (avformat_open_input(&pFormatCtx, url.c_str(), NULL, NULL) != 0) {
    fprintf(stderr, "error loading %s\n", url.c_str());
    emit finished(false);
    return;
  }
  avformat_find_stream_info(pFormatCtx, NULL);
  av_dump_format(pFormatCtx, 0, url.c_str(), 0);

  auto pCodecCtxOrig = pFormatCtx->streams[0]->codec;
  auto pCodec = avcodec_find_decoder(pCodecCtxOrig->codec_id);
  assert(pCodec != NULL);

  pCodecCtx = avcodec_alloc_context3(pCodec);
  int ret = avcodec_copy_context(pCodecCtx, pCodecCtxOrig);
  assert(ret == 0);

  ret = avcodec_open2(pCodecCtx, pCodec, NULL);
  assert(ret >= 0);

  width = pCodecCtxOrig->width;
  height = pCodecCtxOrig->height;

  sws_ctx = sws_getContext(width, height, AV_PIX_FMT_YUV420P,
                           width, height, AV_PIX_FMT_BGR24,
                           SWS_BILINEAR, NULL, NULL, NULL);
  assert(sws_ctx != NULL);

  do {
    Frame &frame = frames.emplace_back();
    if (av_read_frame(pFormatCtx, &frame.pkt) < 0) {
      frames.pop_back();
      break;
    }
  } while (!exit_);

  printf("framereader download done\n");
  valid_ = !exit_;
  emit finished(valid_);
}

void FrameReader::decodeFrames() {
  while (!exit_) {
    while (decode_idx != -1) {
      int from = std::max(decode_idx - decode_idx % 15, 0);
      int to = std::min(from + 15, (int)frames.size());
      decode_idx = -1;

      // the loop will be breaked if another FrameReader::get() is called (decode_idx != -1).
      for (int i = from; i < to && !exit_ && decode_idx == -1; ++i) {
        Frame &frame = frames[i];
        if (frame.picture != nullptr || frame.failed) continue;

        int gotFrame;
        AVFrame *pFrame = av_frame_alloc();
        avcodec_decode_video2(pCodecCtx, pFrame, &gotFrame, &frame.pkt);
        av_free_packet(&frame.pkt);
        
        AVFrame *picture = gotFrame ? toRGB(pFrame) : nullptr;
        av_frame_free(&pFrame);

        if (!picture) {
          qDebug() << "failed to decode frame " << i << " in " << url.c_str();
        }
        std::unique_lock lk(mutex);
        frame.picture = picture;
        frame.failed = !picture;
        cv_frame.notify_all();
      }
    }

    // sleep & wait
    std::unique_lock lk(mutex);
    cv_decode.wait(lk, [=] { return exit_ || decode_idx != -1; });
  }
}

AVFrame *FrameReader::toRGB(AVFrame *pFrame) {
  AVFrame *pFrameRGB = av_frame_alloc();
  int numBytes = avpicture_get_size(AV_PIX_FMT_BGR24, pFrame->width, pFrame->height);
  uint8_t *buffer = (uint8_t *)av_malloc(numBytes * sizeof(uint8_t));
  avpicture_fill((AVPicture *)pFrameRGB, buffer, AV_PIX_FMT_BGR24, pFrame->width, pFrame->height);
  sws_scale(sws_ctx, (uint8_t const *const *)pFrame->data,
            pFrame->linesize, 0, pFrame->height,
            pFrameRGB->data, pFrameRGB->linesize);
  return pFrameRGB;
}

uint8_t *FrameReader::get(int idx) {
  if (!valid_ || idx < 0 || idx >= frames.size()) return nullptr;

  std::unique_lock lk(mutex);
  decode_idx = idx;
  cv_decode.notify_one();
  const Frame &frame = frames[idx];
  if (!frame.picture && !frame.failed) {
    cv_frame.wait(lk, [=] { return exit_ || frame.picture != nullptr || frame.failed; });
  }
  return frame.picture ? frame.picture->data[0] : nullptr;
}
