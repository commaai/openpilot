#include "tools/clib/framereader.h"

#include <assert.h>
#include <unistd.h>

static int ffmpeg_lockmgr_cb(void **arg, enum AVLockOp op) {
  pthread_mutex_t *mutex = (pthread_mutex_t *)*arg;
  int err;

  switch (op) {
  case AV_LOCK_CREATE:
    mutex = (pthread_mutex_t *)malloc(sizeof(*mutex));
    if (!mutex)
        return AVERROR(ENOMEM);
    if ((err = pthread_mutex_init(mutex, NULL))) {
        free(mutex);
        return AVERROR(err);
    }
    *arg = mutex;
    return 0;
  case AV_LOCK_OBTAIN:
    if ((err = pthread_mutex_lock(mutex)))
        return AVERROR(err);

    return 0;
  case AV_LOCK_RELEASE:
    if ((err = pthread_mutex_unlock(mutex)))
        return AVERROR(err);

    return 0;
  case AV_LOCK_DESTROY:
    if (mutex)
        pthread_mutex_destroy(mutex);
    free(mutex);
    *arg = NULL;
    return 0;
  }
  return 1;
}

FrameReader::FrameReader(const char *fn) {
  int ret;

  ret = av_lockmgr_register(ffmpeg_lockmgr_cb);
  assert(ret >= 0);

  avformat_network_init();
  av_register_all();

  snprintf(url, sizeof(url)-1,"%s",fn);
  t = new std::thread([&]() { this->loaderThread(); });
}

void FrameReader::loaderThread() {
  int ret;

  if (avformat_open_input(&pFormatCtx, url, NULL, NULL) != 0) {
    fprintf(stderr, "error loading %s\n", url);
    valid = false;
    return;
  }
  av_dump_format(pFormatCtx, 0, url, 0);

  auto pCodecCtxOrig = pFormatCtx->streams[0]->codec;
  auto pCodec = avcodec_find_decoder(pCodecCtxOrig->codec_id);
  assert(pCodec != NULL);

  pCodecCtx = avcodec_alloc_context3(pCodec);
  ret = avcodec_copy_context(pCodecCtx, pCodecCtxOrig);
  assert(ret == 0);

  ret = avcodec_open2(pCodecCtx, pCodec, NULL);
  assert(ret >= 0);

  sws_ctx = sws_getContext(width, height, AV_PIX_FMT_YUV420P,
                           width, height, AV_PIX_FMT_BGR24,
                           SWS_BILINEAR, NULL, NULL, NULL);
  assert(sws_ctx != NULL);

  AVPacket *pkt = (AVPacket *)malloc(sizeof(AVPacket));
  assert(pkt != NULL);
  bool first = true;
  while (av_read_frame(pFormatCtx, pkt)>=0) {
    //printf("%d pkt %d %d\n", pkts.size(), pkt->size, pkt->pos);
    if (first) {
      AVFrame *pFrame = av_frame_alloc();
      int frameFinished;
      avcodec_decode_video2(pCodecCtx, pFrame, &frameFinished, pkt);
      first = false;
    }
    pkts.push_back(pkt);
    pkt = (AVPacket *)malloc(sizeof(AVPacket));
    assert(pkt != NULL);
  }
  free(pkt);
  printf("framereader download done\n");
  joined = true;

  // cache
  while (1) {
    GOPCache(to_cache.get());
  }
}


void FrameReader::GOPCache(int idx) {
  AVFrame *pFrame;
  int gop = idx - idx%15;

  mcache.lock();
  bool has_gop = cache.find(gop) != cache.end();
  mcache.unlock();

  if (!has_gop) {
    //printf("caching %d\n", gop);
    for (int i = gop; i < gop+15; i++) {
      if (i >= pkts.size()) break;
      //printf("decode %d\n", i);
      int frameFinished;
      pFrame = av_frame_alloc();
      avcodec_decode_video2(pCodecCtx, pFrame, &frameFinished, pkts[i]);
      uint8_t *dat = toRGB(pFrame)->data[0];
      mcache.lock();
      cache.insert(std::make_pair(i, dat));
      mcache.unlock();
    }
  }
}

AVFrame *FrameReader::toRGB(AVFrame *pFrame) {
  AVFrame *pFrameRGB = av_frame_alloc();
  int numBytes = avpicture_get_size(AV_PIX_FMT_BGR24, pFrame->width, pFrame->height);
  uint8_t *buffer = (uint8_t *)av_malloc(numBytes*sizeof(uint8_t));
  avpicture_fill((AVPicture *)pFrameRGB, buffer, AV_PIX_FMT_BGR24, pFrame->width, pFrame->height);
	sws_scale(sws_ctx, (uint8_t const * const *)pFrame->data,
						pFrame->linesize, 0, pFrame->height,
						pFrameRGB->data, pFrameRGB->linesize);
  return pFrameRGB;
}

uint8_t *FrameReader::get(int idx) {
  if (!valid) return NULL;
  waitForReady();
  // TODO: one line?
  uint8_t *dat = NULL;

  // lookahead
  to_cache.put(idx);
  to_cache.put(idx+15);

  mcache.lock();
  auto it = cache.find(idx);
  if (it != cache.end()) {
    dat = it->second;
  }
  mcache.unlock();

  if (dat == NULL) {
    to_cache.put_front(idx);
    // lookahead
    while (dat == NULL) {
      // wait for frame
      usleep(50*1000);
      // check for frame
      mcache.lock();
      auto it = cache.find(idx);
      if (it != cache.end()) dat = it->second;
      mcache.unlock();
      if (dat == NULL) {
        printf(".");
        fflush(stdout);
      }
    }
  }
  return dat;
}

