#pragma once
#include <vector>
#include <string>
#include <unistd.h>

#include "messaging/messaging.h"
#include "visionipc/visionipc.h"
#include "visionipc/visionbuf.h"

class VisionIpcClient {
private:
  std::string name;
  Context * msg_ctx;
  SubSocket * sock;
  Poller * poller;

  VisionStreamType type;

  cl_device_id device_id = nullptr;
  cl_context ctx = nullptr;

  void init_msgq(bool conflate);

public:
  bool connected = false;
  int num_buffers = 0;
  VisionBuf buffers[VISIONIPC_MAX_FDS];
  VisionIpcClient(std::string name, VisionStreamType type, bool conflate, cl_device_id device_id=nullptr, cl_context ctx=nullptr);
  ~VisionIpcClient();
  VisionBuf * recv(VisionIpcBufExtra * extra=nullptr, const int timeout_ms=100);
  bool connect(bool blocking=true);
  bool is_connected() { return connected; }
};
