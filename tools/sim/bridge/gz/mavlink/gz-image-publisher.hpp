#include <CL/cl.h>
#include <cstdint>
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
  std::pair<uint16_t, uint16_t> image_size = std::make_pair(720, 1280);
};

class GZImagePublisher {
public:
  explicit GZImagePublisher(const BridgeConfig config = {});
  ~GZImagePublisher();
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
  std::string last_nv12_frame;
  gz::transport::Node node_;
  cl_platform_id platform;
  cl_device_id device_id;
  cl_context context;
  cl_command_queue command_queue;
  cl_mem cl_input;
  cl_mem cl_output;
  cl_program program;
  cl_kernel kernel;
//  Autopilot_Interface autopilot_interface_;
};
