#include <chrono>
#include <cassert>
#include <iostream>
#include <thread>

#include "visionipc/ipc.h"
#include "visionipc/visionipc_client.h"
#include "visionipc/visionipc_server.h"
#include "logger/logger.h"

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

  // Connect to server socket and ask for all FDs of type
  std::string path = "/tmp/visionipc_" + name;

  int socket_fd = -1;
  while (socket_fd < 0) {
    socket_fd = ipc_connect(path.c_str());

    if (socket_fd < 0) {
      if (blocking){
        std::cout << "VisionIpcClient connecting" << std::endl;
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
      } else {
        return false;
      }
    }
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
