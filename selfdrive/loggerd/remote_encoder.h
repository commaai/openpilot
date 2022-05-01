#include "selfdrive/loggerd/video_writer.h"
#define V4L2_BUF_FLAG_KEYFRAME 0x00000008

struct RemoteEncoder {
  std::unique_ptr<VideoWriter> writer;
  int segment = -1;
  std::vector<Message *> q;
  int logger_segment = -1;
  int dropped_frames = 0;
};

int handle_encoder_msg(LoggerdState *s, Message *msg, std::string &name, struct RemoteEncoder &re);