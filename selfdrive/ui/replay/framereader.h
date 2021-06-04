#pragma once

#include <unistd.h>

#include <atomic>
#include <condition_variable>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <QThread>

// independent of QT, needs ffmpeg
extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libswscale/swscale.h>
}

class FrameReader : public QObject {
  Q_OBJECT

public:
  FrameReader(const std::string &url, QObject *parent = nullptr);
  ~FrameReader();
  bool get(int idx, void* addr);
  bool valid() const { return valid_; }
  // int getRGBSize() const { return width * height * 3; }

  int width = 0, height = 0;

signals:
  void finished(bool success);

private:
  void process();
  bool processFrames();
  void decodeThread();
  bool toRGB(AVFrame *frm, void *addr);

  struct Frame {
    AVPacket pkt = {};
    AVFrame *picture = nullptr;
    bool failed = false;
  };
  std::vector<Frame> frames_;

  AVFormatContext *pFormatCtx_ = NULL;
  AVCodecContext *pCodecCtx_ = NULL;
  AVFrame *frmRgb_ = nullptr;
  struct SwsContext *sws_ctx_ = NULL;

  std::mutex mutex_;
  std::condition_variable cv_decode_;
  std::condition_variable cv_frame_;
  int decode_idx_ = 0;
  std::atomic<bool> exit_ = false;
  bool valid_ = false;
  std::string url_;
  QThread *process_thread_;
  std::thread decode_thread_;
};
