#include "selfdrive/loggerd/encoder/video_writer.h"

struct RemoteEncoder {
  std::unique_ptr<VideoWriter> writer;
  int segment = -1;
  std::vector<Message *> q;
  int logger_segment = -1;
  int dropped_frames = 0;
};

int handle_encoder_msg(LoggerdState *s, Message *msg, std::string &name, struct RemoteEncoder &re);