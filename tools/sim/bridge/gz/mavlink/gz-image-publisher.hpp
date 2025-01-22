#include <fcntl.h>
#include <gz/msgs.hh>
#include <gz/transport.hh>
#include <thread>
#include <unistd.h>

#include "cereal/gen/cpp/car.capnp.h"
#include "cereal/gen/cpp/log.capnp.h"
#include "pid-controller.hpp"
#include <capnp/message.h>

#include <capnp/serialize-packed.h>

bool isFileDescriptiorValid(int client_socket_fd) {
  return fcntl(client_socket_fd, F_GETFD) != -1 || errno != EBADF;
}

struct BridgeConfig {
  uint16_t port = 4069;
};

class GZImagePublisher {
public:
  explicit GZImagePublisher(const BridgeConfig config = {});
  ~GZImagePublisher() {
    if (isFileDescriptiorValid(log_file_fd_)) {
      close(log_file_fd_);
    }
  };
  void run();

private:
  void send_image(int);
  void on_image(const gz::msgs::Image &_msg);
  bool sub_camera(const std::string);
  void run_tcp_server(const uint16_t port);
  int log_file_fd_ = -1;
  float last_angle = 0;
  gz::msgs::Image last_frame;
  BridgeConfig config_;
  PIDController pid_controller_yaw_;
  std::thread tcp_server_thread_;
  gz::transport::Node node_;
//  Autopilot_Interface autopilot_interface_;
};
