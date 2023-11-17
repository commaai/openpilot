#pragma once

#include <set>
#include <string>
#include <vector>
#include <unistd.h>

#include "cereal/messaging/messaging.h"
#include "cereal/visionipc/visionipc.h"
#include "cereal/visionipc/visionbuf.h"

class VisionIpcClient {
private:
  std::string name;
  Context * msg_ctx;
  SubSocket * sock;
  Poller * poller;

  cl_device_id device_id = nullptr;
  cl_context ctx = nullptr;

public:
  bool connected = false;
  VisionStreamType type;
  int num_buffers = 0;
  VisionBuf buffers[VISIONIPC_MAX_FDS];
  VisionIpcClient(std::string name, VisionStreamType type, bool conflate, cl_device_id device_id=nullptr, cl_context ctx=nullptr);
  ~VisionIpcClient();
  VisionBuf * recv(VisionIpcBufExtra * extra=nullptr, const int timeout_ms=100);
  bool connect(bool blocking=true);
  bool is_connected() { return connected; }
  static std::set<VisionStreamType> getAvailableStreams(const std::string &name, bool blocking = true);
};
