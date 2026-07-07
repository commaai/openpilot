#include "tools/cabana/streams/socketcanstream.h"

#include <linux/can.h>
#include <linux/can/raw.h>
#include <net/if.h>
#include <sys/ioctl.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cstdio>
#include <cstring>

SocketCanStream::SocketCanStream(SocketCanStreamConfig config_) : config(config_) {
  if (!available()) {
    throw std::runtime_error("SocketCAN not available");
  }

  fprintf(stderr, "Connecting to SocketCAN device %s\n", config.device.c_str());
  if (!connect()) {
    throw std::runtime_error("Failed to connect to SocketCAN device");
  }
}

SocketCanStream::~SocketCanStream() {
  stop();
  if (sock_fd >= 0) {
    ::close(sock_fd);
    sock_fd = -1;
  }
}

bool SocketCanStream::available() {
  int fd = socket(PF_CAN, SOCK_RAW, CAN_RAW);
  if (fd < 0) return false;
  ::close(fd);
  return true;
}

bool SocketCanStream::connect() {
  sock_fd = socket(PF_CAN, SOCK_RAW, CAN_RAW);
  if (sock_fd < 0) {
    fprintf(stderr, "Failed to create CAN socket\n");
    return false;
  }

  // Enable CAN-FD
  int fd_enable = 1;
  setsockopt(sock_fd, SOL_CAN_RAW, CAN_RAW_FD_FRAMES, &fd_enable, sizeof(fd_enable));

  struct ifreq ifr = {};
  strncpy(ifr.ifr_name, config.device.c_str(), IFNAMSIZ - 1);
  if (ioctl(sock_fd, SIOCGIFINDEX, &ifr) < 0) {
    fprintf(stderr, "Failed to get interface index for %s\n", config.device.c_str());
    ::close(sock_fd);
    sock_fd = -1;
    return false;
  }

  struct sockaddr_can addr = {};
  addr.can_family = AF_CAN;
  addr.can_ifindex = ifr.ifr_ifindex;
  if (bind(sock_fd, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
    fprintf(stderr, "Failed to bind CAN socket\n");
    ::close(sock_fd);
    sock_fd = -1;
    return false;
  }

  // Set read timeout so the thread can check for interruption
  struct timeval tv = {.tv_sec = 0, .tv_usec = 100000};  // 100ms
  setsockopt(sock_fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));

  return true;
}

void SocketCanStream::streamThread() {
  struct canfd_frame frame;

  while (!stop_requested_) {
    ssize_t nbytes = read(sock_fd, &frame, sizeof(frame));
    if (nbytes <= 0) continue;

    uint8_t len = (nbytes == CAN_MTU) ? frame.len : frame.len;  // works for both CAN and CAN-FD

    MessageBuilder msg;
    auto evt = msg.initEvent();
    auto canData = evt.initCan(1);
    canData[0].setAddress(frame.can_id & CAN_EFF_MASK);
    canData[0].setSrc(0);
    canData[0].setDat(kj::arrayPtr(frame.data, len));

    handleEvent(capnp::messageToFlatArray(msg));
  }
}
