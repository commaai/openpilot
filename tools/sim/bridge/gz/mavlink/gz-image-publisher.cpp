//
// Example that demonstrates offboard control using attitude, velocity control
// in NED (North-East-Down), and velocity control in body (Forward-Right-Down)
// coordinates.
//

#include <asm-generic/socket.h>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <ctime>
#include <fcntl.h>
#include <future>
#include <iostream>
#include <memory>
#include <signal.h>
#include <sstream>
#include <thread>
#include <unistd.h>

#include "gz-image-publisher.hpp"
#include <netinet/in.h>
#include <netinet/ip.h>
#include <sys/socket.h>

using std::chrono::milliseconds;
using std::chrono::seconds;
using std::this_thread::sleep_for;

GZImagePublisher::GZImagePublisher(const BridgeConfig config)
    : config_(config), pid_controller_yaw_(4, 0.3, 1.5) {}

bool GZImagePublisher::sub_camera(const std::string topic) {
  return this->node_.Subscribe(
      topic, std::function<void(const gz::msgs::Image &_msg)>(std::bind(
                 &GZImagePublisher::on_image, this, std::placeholders::_1)));
}

void GZImagePublisher::on_image(const gz::msgs::Image &_msg) {
  this->last_frame = gz::msgs::Image(_msg);
}

void GZImagePublisher::run() {
  std::cout << "Starting MavlinkBridge" << std::endl;
  std::cout << "Connecting to gazebo sim" << std::endl;
  if (!this->sub_camera("/camera")) {
    throw std::runtime_error("Failed to subscribe to camera topic");
  }
  std::cout << "Starting tcp server" << std::endl;
  this->log_file_fd_ = open("/tmp/mavlink-bridge.csv",
                            O_WRONLY | O_CREAT | O_TRUNC, S_IRUSR | S_IWUSR);
  if (this->log_file_fd_ < 0) {
    throw std::runtime_error("Failed to open log file");
  }
  std::string header =
      "time,gas,break,front_speed,steering,yaw_speed_deg,yaw_speed_deg\n";
  write(this->log_file_fd_, header.c_str(), header.size());
  this->run_tcp_server(this->config_.port);
}

void GZImagePublisher::run_tcp_server(const uint16_t port) {
  int tcp_socket = socket(AF_INET, SOCK_STREAM, 0);
  if (tcp_socket < 0) {
    throw std::runtime_error("Failed to create TCP socket");
  }
  const int optval = 1;
  setsockopt(tcp_socket, SOL_SOCKET, SO_REUSEADDR, &optval, sizeof(optval));
  sockaddr_in server_addr{.sin_family = AF_INET,
                          .sin_port = htons(port),
                          .sin_addr =
                              {
                                  .s_addr = INADDR_ANY,
                              },
                          .sin_zero = {0, 0, 0, 0, 0, 0, 0, 0}};
  if (bind(tcp_socket, reinterpret_cast<sockaddr *>(&server_addr),
           sizeof(server_addr)) < 0) {
    throw std::runtime_error("Failed to bind TCP socket");
  };
  if (listen(tcp_socket, 10) < 0) {
    throw std::runtime_error("Failed to listen on TCP socket");
  }
  std::cout << "Listening on port " << port << std::endl;
  signal(SIGPIPE, SIG_IGN);
  while (true) {
    sockaddr_in client_addr{};
    socklen_t client_addr_len = sizeof(client_addr);
    int client_socket_fd =
        accept(tcp_socket, reinterpret_cast<sockaddr *>(&client_addr),
               &client_addr_len);
    if (client_socket_fd < 0) {
      throw std::runtime_error("Failed to accept TCP connection");
    }
    uint8_t buf[1];
    auto bytes_read = recv(client_socket_fd, buf, sizeof(uint8_t), MSG_WAITALL);
    if (bytes_read <= 0) {
      continue;
    }
    std::thread client_thread([this, client_socket_fd, buf]() {
      try {
        if (buf[0] == 0x00) {
          this->send_image(client_socket_fd);
        }
      } catch (...) {
        std::cerr << "Failed to send image" << std::endl;
      }
      if (isFileDescriptiorValid(client_socket_fd)) {
        shutdown(client_socket_fd, SHUT_RDWR);
        close(client_socket_fd);
      }
    });
    client_thread.detach();
  }
}

void GZImagePublisher::send_image(int client_socket_fd) {
  ::capnp::MallocMessageBuilder message;
  cereal::Thumbnail::Builder thumb = message.initRoot<cereal::Thumbnail>();
  thumb.setFrameId(0);
  thumb.setTimestampEof(0);
  kj::ArrayPtr<const uint8_t> data(
      (const unsigned char *)this->last_frame.data().data(),
      this->last_frame.data().size());
  thumb.setThumbnail(data);
  thumb.setEncoding(cereal::Thumbnail::Encoding::UNKNOWN);
  if (isFileDescriptiorValid(client_socket_fd)) {
    writePackedMessageToFd(client_socket_fd, message);
  }
}

int main() {
  BridgeConfig config = {};
  GZImagePublisher bridge(config);
  bridge.run();
  return 0;
}
