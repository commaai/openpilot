#include "selfdrive/ui/replay/framereader.h"

#include <assert.h>
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

FrameReader::FrameReader(const std::string &fn) : url(fn) {
  int ret = av_lockmgr_register(ffmpeg_lockmgr_cb);
  assert(ret >= 0);

  avformat_network_init();
  av_register_all();
}

FrameReader::~FrameReader() {
  exit_ = true;
  thread.join();
  for (auto &f : frames) {
    delete f->pkt;
    if (f->picture) {
      av_frame_free(&f->picture);
    }
    delete f;
  }
  avcodec_free_context(&pCodecCtx);
  avformat_free_context(pFormatCtx);
  sws_freeContext(sws_ctx);
  avformat_network_deinit();
}

void FrameReader::process() {
  if (avformat_open_input(&pFormatCtx, url.c_str(), NULL, NULL) != 0) {
    fprintf(stderr, "error loading %s\n", url.c_str());
    valid = false;
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
    AVPacket *pkt = new AVPacket;
    if (av_read_frame(pFormatCtx, pkt) < 0) {
      delete pkt;
      break;
    }
    Frame *frame = new Frame;
    frame->pkt = pkt;
    frames.push_back(frame);
  } while (true);

  printf("framereader download done\n");

  thread = std::thread(&FrameReader::decodeThread, this);
}

void FrameReader::decodeThread() {
  while (!exit_) {
    int gop = 0;
    {
      std::unique_lock lk(mutex);
      cv_decode.wait(lk, [=] { return exit_ || decode_idx != -1; });
      if (exit_) break;

      gop = std::max(decode_idx - decode_idx % 15, 0);
      decode_idx = -1;
    }

    for (int i = gop; i < std::min(gop + 15, (int)frames.size()); ++i) {
      if (frames[i]->picture != nullptr) continue;

      int frameFinished;
      AVFrame *pFrame = av_frame_alloc();
      avcodec_decode_video2(pCodecCtx, pFrame, &frameFinished, frames[i]->pkt);
      AVFrame *picture = toRGB(pFrame);
      av_frame_free(&pFrame);

      std::unique_lock lk(mutex);
      frames[i]->picture = picture;
      cv_frame.notify_all();
    }
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
  if (!valid || idx < 0 || idx >= frames.size()) return nullptr;

  std::unique_lock lk(mutex);
  decode_idx = idx;
  cv_decode.notify_one();
  Frame *frame = frames[idx];
  if (!frame->picture) {
    cv_frame.wait(lk, [=] { return exit_ || frame->picture != nullptr; });
  }
  return frame->picture ? frame->picture->data[0] : nullptr;
}
