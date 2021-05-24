#pragma once

#include <unistd.h>
#include <vector>
#include <map>
#include <thread>
#include <mutex>
#include <list>
#include <condition_variable>

#include <QString>

#include "tools/clib/channel.h"

// independent of QT, needs ffmpeg
extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libswscale/swscale.h>
}


class FrameReader {
public:
  FrameReader(const QString &fn);
  uint8_t *get(int idx);
  AVFrame *toRGB(AVFrame *);
  void waitForReady() {
    while (!joined) usleep(10*1000);
  }
  int getRGBSize() { return width*height*3; }
  void process();

  //TODO: get this from the actual frame
  int width = 1164;
  int height = 874;

private:
  AVFormatContext *pFormatCtx = NULL;
  AVCodecContext *pCodecCtx = NULL;

	struct SwsContext *sws_ctx = NULL;

  std::vector<AVPacket *> pkts;

  bool joined = false;

  std::map<int, uint8_t *> cache;
  std::mutex mcache;

  void GOPCache(int idx);
  channel<int> to_cache;

  bool valid = true;
  QString url;
};

