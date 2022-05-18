#include <iostream>
#include <chrono>
#include <cassert>
#include <random>

#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>

#include "messaging/messaging.h"
#include "visionipc/ipc.h"
#include "visionipc/visionipc_server.h"
#include "logger/logger.h"

std::string get_endpoint_name(std::string name, VisionStreamType type){
  if (messaging_use_zmq()){
    assert(name == "camerad" || name == "navd");
    return std::to_string(9000 + static_cast<int>(type));
  } else {
    return "visionipc_" + name + "_" + std::to_string(type);
  }
}

VisionIpcServer::VisionIpcServer(std::string name, cl_device_id device_id, cl_context ctx) : name(name), device_id(device_id), ctx(ctx) {
  msg_ctx = Context::create();

  std::random_device rd("/dev/urandom");
  std::uniform_int_distribution<uint64_t> distribution(0,std::numeric_limits<uint64_t>::max());
  server_id = distribution(rd);
}

void VisionIpcServer::create_buffers(VisionStreamType type, size_t num_buffers, bool rgb, size_t width, size_t height){
  // TODO: assert that this type is not created yet
  assert(num_buffers < VISIONIPC_MAX_FDS);
  int aligned_w = 0, aligned_h = 0;

  size_t size = 0;
  size_t stride = 0; // Only used for RGB

  if (rgb) {
    visionbuf_compute_aligned_width_and_height(width, height, &aligned_w, &aligned_h);
    size = (size_t)aligned_w * (size_t)aligned_h * 3;
    stride = aligned_w * 3;
  } else {
    size = width * height * 3 / 2;
  }

  // Create map + alloc requested buffers
  for (size_t i = 0; i < num_buffers; i++){
    VisionBuf* buf = new VisionBuf();
    buf->allocate(size);
    buf->idx = i;
    buf->type = type;

    if (device_id) buf->init_cl(device_id, ctx);

    rgb ? buf->init_rgb(width, height, stride) : buf->init_yuv(width, height);

    buffers[type].push_back(buf);
  }

  cur_idx[type] = 0;

  // Create msgq publisher for each of the `name` + type combos
  // TODO: compute port number directly if using zmq
  sockets[type] = PubSocket::create(msg_ctx, get_endpoint_name(name, type), false);
}


void VisionIpcServer::start_listener(){
  listener_thread = std::thread(&VisionIpcServer::listener, this);
}


void VisionIpcServer::listener(){
  std::cout << "Starting listener for: " << name << std::endl;

  std::string path = "/tmp/visionipc_" + name;
  int sock = ipc_bind(path.c_str());
  assert(sock >= 0);

  while (!should_exit){
    // Wait for incoming connection
    struct pollfd polls[1] = {{0}};
    polls[0].fd = sock;
    polls[0].events = POLLIN;

    int ret = poll(polls, 1, 100);
    if (ret < 0) {
      if (errno == EINTR || errno == EAGAIN) continue;
      std::cout << "poll failed, stopping listener" << std::endl;
      break;
    }

    if (should_exit) break;
    if (!polls[0].revents) {
      continue;
    }

    // Handle incoming request
    int fd = accept(sock, NULL, NULL);
    assert(fd >= 0);

    VisionStreamType type = VisionStreamType::VISION_STREAM_MAX;
    int r = ipc_sendrecv_with_fds(false, fd, &type, sizeof(type), nullptr, 0, nullptr);
    assert(r == sizeof(type));
    if (buffers.count(type) <= 0) {
      std::cout << "got request for invalid buffer type: " << type << std::endl;
      close(fd);
      continue;
    }

    int fds[VISIONIPC_MAX_FDS];
    int num_fds = buffers[type].size();
    VisionBuf bufs[VISIONIPC_MAX_FDS];

    for (int i = 0; i < num_fds; i++){
      fds[i] = buffers[type][i]->fd;
      bufs[i] = *buffers[type][i];

      // Remove some private openCL/ion metadata
      bufs[i].buf_cl = 0;
      bufs[i].copy_q = 0;
      bufs[i].handle = 0;

      bufs[i].server_id = server_id;
    }

    r = ipc_sendrecv_with_fds(true, fd, &bufs, sizeof(VisionBuf) * num_fds, fds, num_fds, nullptr);

    close(fd);
  }

  std::cout << "Stopping listener for: " << name << std::endl;
  close(sock);
}



VisionBuf * VisionIpcServer::get_buffer(VisionStreamType type){
  // Do we want to keep track if the buffer has been sent out yet and warn user?
  assert(buffers.count(type));
  auto b = buffers[type];
  return b[cur_idx[type]++ % b.size()];
}

void VisionIpcServer::send(VisionBuf * buf, VisionIpcBufExtra * extra, bool sync){
  if (sync) {
    if (buf->sync(VISIONBUF_SYNC_FROM_DEVICE) != 0) {
      LOGE("Failed to sync buffer");
    }
  }
  assert(buffers.count(buf->type));
  assert(buf->idx < buffers[buf->type].size());

  // Send over correct msgq socket
  VisionIpcPacket packet = {0};
  packet.server_id = server_id;
  packet.idx = buf->idx;
  packet.extra = *extra;

  sockets[buf->type]->send((char*)&packet, sizeof(packet));
}

VisionIpcServer::~VisionIpcServer(){
  should_exit = true;
  listener_thread.join();

  // VisionBuf cleanup
  for( auto const& [type, buf] : buffers ) {
    for (VisionBuf* b : buf){
      if (b->free() != 0) {
        LOGE("Failed to free buffer");
      }
      delete b;
    }
  }

  // Messaging cleanup
  for( auto const& [type, sock] : sockets ) {
    delete sock;
  }
  delete msg_ctx;
}
