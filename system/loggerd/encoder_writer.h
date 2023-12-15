#pragma once

#include <deque>
#include <memory>
#include <string>

#include "system/loggerd/logger.h"
#include "system/loggerd/loggerd.h"
#include "system/loggerd/video_writer.h"

class EncoderWriter {
public:
  EncoderWriter(const std::string &path, const EncoderInfo &info) : segment_path(path), info(info) {}
  void rotate(const std::string &path);
  size_t write(LoggerState *logger, Message *msg);
  size_t flush(LoggerState *logger);
  static int readyToRotate() { return ready_to_rotate; }

private:
  size_t write_encoder_data(LoggerState *logger, const cereal::Event::Reader event);
  size_t write_video(const cereal::EncodeData::Reader &edata, const cereal::EncodeIndex::Reader &idx);

  std::unique_ptr<VideoWriter> video_writer;
  int remote_encoder_segment = -1;
  int current_encoder_segment = -1;
  std::string segment_path;
  std::deque<std::unique_ptr<Message>> q;
  int dropped_frames = 0;
  bool marked_ready_to_rotate = false;
  EncoderInfo info;

  inline static int ready_to_rotate = 0;  // count of encoders ready to rotate
};
