#pragma once

#include <unistd.h>

#include <atomic>
#include <condition_variable>
#include <mutex>
#include <thread>
#include <vector>
#include "cereal/visionipc/visionbuf.h"

#include <QThread>

// independent of QT, needs ffmpeg
extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libswscale/swscale.h>
}

class FrameReader : public QThread {
  Q_OBJECT

public:
  FrameReader(const std::string &fn, VisionStreamType stream_type, QObject *parent);
  ~FrameReader();
  void run() override;
  uint8_t *get(int idx);
  bool valid() const {return valid_;}
  int getRGBSize() const { return width*height*3; }

  int width = 0, height = 0;
  VisionStreamType stream_type;

signals:
  void finished(bool success);

private:
  void process();
  void decodeFrames();
  AVFrame *toRGB(AVFrame *);

  struct Frame {
    AVPacket pkt;
    AVFrame *picture;
  };
  std::vector<Frame*> frames;

  AVFormatContext *pFormatCtx = NULL;
  AVCodecContext *pCodecCtx = NULL;
	struct SwsContext *sws_ctx = NULL;

  std::mutex mutex;
  std::condition_variable cv_decode;
  std::condition_variable cv_frame;
  std::atomic<int> decode_idx = 0;
  std::atomic<bool> exit_ = false;

  bool valid_ = true;
  std::string url;
};
