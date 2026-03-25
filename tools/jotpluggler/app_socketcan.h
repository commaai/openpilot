#pragma once

#include "tools/jotpluggler/jotpluggler.h"

#include <string>
#include <vector>

std::vector<std::string> list_socketcan_devices();

class SocketCanReader {
public:
  explicit SocketCanReader(const std::string &device);
  ~SocketCanReader();

  SocketCanReader(const SocketCanReader &) = delete;
  SocketCanReader &operator=(const SocketCanReader &) = delete;

  bool readFrame(LiveCanFrame *frame);

private:
  int sock_fd_ = -1;
};
