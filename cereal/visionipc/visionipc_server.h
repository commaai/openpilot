#pragma once
#include <vector>
#include <string>
#include <thread>
#include <atomic>
#include <map>

#include "messaging/messaging.h"
#include "visionipc/visionipc.h"
#include "visionipc/visionbuf.h"

std::string get_endpoint_name(std::string name, VisionStreamType type);

class VisionIpcServer {
 private:
  cl_device_id device_id = nullptr;
  cl_context ctx = nullptr;
  uint64_t server_id;

  std::atomic<bool> should_exit = false;
  std::string name;
  std::thread listener_thread;

  std::map<VisionStreamType, std::atomic<size_t> > cur_idx;
  std::map<VisionStreamType, std::vector<VisionBuf*> > buffers;
  std::map<VisionStreamType, std::map<VisionBuf*, size_t> > idxs;

  Context * msg_ctx;
  std::map<VisionStreamType, PubSocket*> sockets;

  void listener(void);

 public:
  VisionIpcServer(std::string name, cl_device_id device_id=nullptr, cl_context ctx=nullptr);
  ~VisionIpcServer();

  VisionBuf * get_buffer(VisionStreamType type);

  void create_buffers(VisionStreamType type, size_t num_buffers, bool rgb, size_t width, size_t height);
  void send(VisionBuf * buf, VisionIpcBufExtra * extra, bool sync=true);
  void start_listener();
};
