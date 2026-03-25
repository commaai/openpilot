#include "tools/jotpluggler/app_socketcan.h"

#include "common/timing.h"

#include <algorithm>
#include <cstdio>
#include <fstream>
#include <stdexcept>

#ifdef __linux__
#include <linux/can.h>
#include <linux/can/raw.h>
#include <net/if.h>
#include <sys/ioctl.h>
#include <sys/socket.h>
#include <unistd.h>
#endif

std::vector<std::string> list_socketcan_devices() {
#ifdef __linux__
  std::vector<std::string> devices;
  const std::filesystem::path net_dir("/sys/class/net");
  std::error_code ec;
  for (const std::filesystem::directory_entry &entry : std::filesystem::directory_iterator(net_dir, ec)) {
    if (ec) break;
    if (!entry.is_directory()) continue;
    std::ifstream type_file(entry.path() / "type");
    int type = -1;
    if (!(type_file >> type) || type != 280) continue;
    devices.push_back(entry.path().filename().string());
  }
  std::sort(devices.begin(), devices.end());
  return devices;
#else
  return {};
#endif
}

SocketCanReader::SocketCanReader(const std::string &device) {
#ifdef __linux__
  sock_fd_ = socket(PF_CAN, SOCK_RAW, CAN_RAW);
  if (sock_fd_ < 0) {
    throw std::runtime_error("Failed to create CAN socket");
  }

  int fd_enable = 1;
  setsockopt(sock_fd_, SOL_CAN_RAW, CAN_RAW_FD_FRAMES, &fd_enable, sizeof(fd_enable));

  struct ifreq ifr = {};
  std::snprintf(ifr.ifr_name, sizeof(ifr.ifr_name), "%s", device.c_str());
  if (ioctl(sock_fd_, SIOCGIFINDEX, &ifr) < 0) {
    ::close(sock_fd_);
    sock_fd_ = -1;
    throw std::runtime_error("Failed to get interface index for " + device);
  }

  struct sockaddr_can addr = {};
  addr.can_family = AF_CAN;
  addr.can_ifindex = ifr.ifr_ifindex;
  if (bind(sock_fd_, reinterpret_cast<struct sockaddr *>(&addr), sizeof(addr)) < 0) {
    ::close(sock_fd_);
    sock_fd_ = -1;
    throw std::runtime_error("Failed to bind CAN socket");
  }

  struct timeval tv = {.tv_sec = 0, .tv_usec = 100000};
  setsockopt(sock_fd_, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
#else
  (void)device;
  throw std::runtime_error("SocketCAN is not available on this platform");
#endif
}

SocketCanReader::~SocketCanReader() {
#ifdef __linux__
  if (sock_fd_ >= 0) {
    ::close(sock_fd_);
    sock_fd_ = -1;
  }
#endif
}

bool SocketCanReader::readFrame(LiveCanFrame *frame) {
#ifdef __linux__
  if (frame == nullptr || sock_fd_ < 0) {
    return false;
  }
  struct canfd_frame can_frame = {};
  const ssize_t nbytes = ::read(sock_fd_, &can_frame, sizeof(can_frame));
  if (nbytes <= 0) {
    return false;
  }
  *frame = LiveCanFrame{
    .mono_time = static_cast<double>(nanos_since_boot()) / 1.0e9,
    .bus = 0,
    .address = can_frame.can_id & CAN_EFF_MASK,
    .bus_time = 0,
    .data = std::string(reinterpret_cast<const char *>(can_frame.data), can_frame.len),
  };
  return true;
#else
  (void)frame;
  return false;
#endif
}
