#include <chrono>
#include <cassert>
#include <iostream>
#include <thread>

#include <unistd.h>
#include "msgq/visionipc/visionipc.h"
#include "msgq/visionipc/visionipc_client.h"
#include "msgq/visionipc/visionipc_server.h"
#include "logger/logger.h"
#include "logger/logger.h"

static int connect_to_vipc_server(const std::string &name, bool blocking) {
  const std::string ipc_path = get_ipc_path(name);
  int socket_fd = ipc_connect(ipc_path.c_str());
  while (socket_fd < 0 && blocking) {
    std::cout << "VisionIpcClient connecting" << std::endl;
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    socket_fd = ipc_connect(ipc_path.c_str());
  }
  return socket_fd;
}

VisionIpcClient::VisionIpcClient(std::string name, VisionStreamType type, bool conflate, cl_device_id device_id, cl_context ctx) : name(name), type(type), device_id(device_id), ctx(ctx) {
  msg_ctx = Context::create();
  sock = SubSocket::create(msg_ctx, get_endpoint_name(name, type), "127.0.0.1", conflate, false);

  poller = Poller::create();
  poller->registerSocket(sock);
}

// Connect is not thread safe. Do not use the buffers while calling connect
bool VisionIpcClient::connect(bool blocking){
  connected = false;

  // Cleanup old buffers on reconnect
  for (size_t i = 0; i < num_buffers; i++){
    if (buffers[i].free() != 0) {
      LOGE("Failed to free buffer %zu", i);
    }
  }

  num_buffers = 0;

  int socket_fd = connect_to_vipc_server(name, blocking);
  if (socket_fd < 0) {
    return false;
  }
  // Send stream type to server to request FDs
  int r = ipc_sendrecv_with_fds(true, socket_fd, &type, sizeof(type), nullptr, 0, nullptr);
  assert(r == sizeof(type));

  // Get FDs
  int fds[VISIONIPC_MAX_FDS];
  VisionBuf bufs[VISIONIPC_MAX_FDS];
  r = ipc_sendrecv_with_fds(false, socket_fd, &bufs, sizeof(bufs), fds, VISIONIPC_MAX_FDS, &num_buffers);

  assert(num_buffers >= 0);
  assert(r == sizeof(VisionBuf) * num_buffers);

  // Import buffers
  for (size_t i = 0; i < num_buffers; i++){
    buffers[i] = bufs[i];
    buffers[i].fd = fds[i];
    buffers[i].import();
    if (buffers[i].rgb) {
      buffers[i].init_rgb(buffers[i].width, buffers[i].height, buffers[i].stride);
    } else {
      buffers[i].init_yuv(buffers[i].width, buffers[i].height, buffers[i].stride, buffers[i].uv_offset);
    }

    if (device_id) buffers[i].init_cl(device_id, ctx);
  }

  close(socket_fd);
  connected = true;
  return true;
}

VisionBuf * VisionIpcClient::recv(VisionIpcBufExtra * extra, const int timeout_ms){
  auto p = poller->poll(timeout_ms);

  if (!p.size()){
    return nullptr;
  }

  Message * r = sock->receive(true);
  if (r == nullptr){
    return nullptr;
  }

  // Get buffer
  assert(r->getSize() == sizeof(VisionIpcPacket));
  VisionIpcPacket *packet = (VisionIpcPacket*)r->getData();

  assert(packet->idx < num_buffers);
  VisionBuf * buf = &buffers[packet->idx];

  if (buf->server_id != packet->server_id){
    connected = false;
    delete r;
    return nullptr;
  }

  if (extra) {
    *extra = packet->extra;
  }

  if (buf->sync(VISIONBUF_SYNC_TO_DEVICE) != 0) {
    LOGE("Failed to sync buffer");
  }

  delete r;
  return buf;
}

std::set<VisionStreamType> VisionIpcClient::getAvailableStreams(const std::string &name, bool blocking) {
  int socket_fd = connect_to_vipc_server(name, blocking);
  if (socket_fd < 0) {
    return {};
  }
  // Send VISION_STREAM_MAX to server to request available streams
  int request = VISION_STREAM_MAX;
  int r = ipc_sendrecv_with_fds(true, socket_fd, &request, sizeof(request), nullptr, 0, nullptr);
  assert(r == sizeof(request));

  VisionStreamType available_streams[VISION_STREAM_MAX] = {};
  r = ipc_sendrecv_with_fds(false, socket_fd, &available_streams, sizeof(available_streams), nullptr, 0, nullptr);
  assert((r >= 0) && (r % sizeof(VisionStreamType) == 0));
  close(socket_fd);
  return std::set<VisionStreamType>(available_streams, available_streams + r / sizeof(VisionStreamType));
}

VisionIpcClient::~VisionIpcClient(){
  for (size_t i = 0; i < num_buffers; i++){
    if (buffers[i].free() != 0) {
      LOGE("Failed to free buffer %zu", i);
    }
  }

  delete sock;
  delete poller;
  delete msg_ctx;
}
