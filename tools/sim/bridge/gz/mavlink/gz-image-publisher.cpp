//
// Example that demonstrates offboard control using attitude, velocity control
// in NED (North-East-Down), and velocity control in body (Forward-Right-Down)
// coordinates.
//

#include <CL/cl.h>
#include <CL/cl_ext.h>
#include <asm-generic/socket.h>
#include <assert.h>
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

void resize_image(const unsigned char *src, unsigned char *dst, int width,
                  int height, int dst_width, int dst_height) {

  float x_ratio = (float)width / dst_width;
  float y_ratio = (float)height / dst_height;

  for (int y = 0; y < dst_height; y++) {
    for (int x = 0; x < dst_width; x++) {
      int src_x = int(x * x_ratio);
      int src_y = int(y * y_ratio);
      size_t dst_pixel = y * dst_width + x;
      size_t src_pixel = src_y * width + src_x;
      for (int c = 0; c < 3; c++) {
        dst[dst_pixel * 3 + c] = src[src_pixel * 3 + c];
      }
    }
  }
}

GZImagePublisher::GZImagePublisher(const BridgeConfig config)
    : config_(config), pid_controller_yaw_(4, 0.3, 1.5),
      last_nv12_frame(
          config_.image_size.first * config_.image_size.second * 3 / 2, ' ') {
  cl_int CL_err = CL_SUCCESS;
  cl_uint numPlatforms = 0;
  cl_platform_id platform;
  cl_device_id device_id;
  CL_err = clGetPlatformIDs(1, &platform, &numPlatforms);
  assert(CL_err == CL_SUCCESS);
  this->platform = platform;

  CL_err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 1, &device_id, NULL);
  assert(CL_err == CL_SUCCESS);
  this->device_id = device_id;

  this->context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &CL_err);
  assert(CL_err == CL_SUCCESS);

  this->command_queue =
      clCreateCommandQueueWithProperties(this->context, device_id, 0, &CL_err);
  assert(CL_err == CL_SUCCESS);

  size_t height = this->config_.image_size.first;
  size_t width = this->config_.image_size.second;
  size_t input_size = height * width * 3;
  size_t output_size = input_size / 2;
  this->cl_input =
      clCreateBuffer(this->context, CL_MEM_READ_ONLY | CL_MEM_HOST_WRITE_ONLY,
                     input_size, NULL, &CL_err);
  assert(CL_err == CL_SUCCESS);
  this->cl_output = clCreateBuffer(this->context, CL_MEM_WRITE_ONLY,
                                   output_size, NULL, &CL_err);
  assert(CL_err == CL_SUCCESS);

  int fd = open("rgb_to_nv12.cl", O_RDONLY);
  assert(fd >= 0);
  size_t size = lseek(fd, 0, SEEK_END);
  lseek(fd, 0, SEEK_SET);
  unsigned char *program_source = new unsigned char[size];
  read(fd, program_source, size);
  close(fd);
  this->program = clCreateProgramWithSource(
      this->context, 1, (const char **)&program_source, &size, nullptr);

  char cl_arg[1024];
  sprintf(cl_arg,
          " -DHEIGHT=%ld -DWIDTH=%ld -DRGB_STRIDE=%ld -DUV_WIDTH=%ld "
          "-DUV_HEIGHT=%ld -DRGB_SIZE=%ld -DCL_DEBUG ",
          height, width, width * 3, width / 2, height / 2, height * width);

  clBuildProgram(this->program, 1, &device_id, cl_arg, nullptr, nullptr);
  this->kernel = clCreateKernel(this->program, "rgb_to_nv12", &CL_err);
  assert(CL_err == CL_SUCCESS);

  clSetKernelArg(this->kernel, 0, sizeof(cl_mem), &this->cl_input);
  clSetKernelArg(this->kernel, 1, sizeof(cl_mem), &this->cl_output);
  delete[] program_source;
}

GZImagePublisher::~GZImagePublisher() {
  clReleaseKernel(this->kernel);
  clReleaseProgram(this->program);
  clReleaseMemObject(this->cl_input);
  clReleaseMemObject(this->cl_output);
  clReleaseCommandQueue(this->command_queue);
  clReleaseContext(this->context);
  if (isFileDescriptiorValid(log_file_fd_)) {
    close(log_file_fd_);
  }
}

bool GZImagePublisher::sub_camera(const std::string topic) {
  return this->node_.Subscribe(
      topic, std::function<void(const gz::msgs::Image &_msg)>(std::bind(
                 &GZImagePublisher::on_image, this, std::placeholders::_1)));
}

void GZImagePublisher::on_image(const gz::msgs::Image &_msg) {
  this->last_frame = gz::msgs::Image(_msg);
  std::string resized_frame(this->last_nv12_frame.size() * 2, ' ');
  resize_image((const unsigned char *)this->last_frame.data().data(),
               (unsigned char *)resized_frame.data(), this->last_frame.width(),
               this->last_frame.height(), this->config_.image_size.second,
               this->config_.image_size.first);
  clEnqueueWriteBuffer(this->command_queue, this->cl_input, CL_TRUE, 0,
                       resized_frame.size(), (void *)resized_frame.data(), 0,
                       nullptr, nullptr);
  size_t global_work_size[2] = {
      static_cast<size_t>(this->config_.image_size.second / 4),
      static_cast<size_t>(this->config_.image_size.first / 4),
  };
  clSetKernelArg(this->kernel, 0, sizeof(cl_mem), &this->cl_input);
  clSetKernelArg(this->kernel, 1, sizeof(cl_mem), &this->cl_output);
  clFinish(this->command_queue);
  cl_event event;
  cl_int CL_err =
      clEnqueueNDRangeKernel(this->command_queue, this->kernel, 2, 0,
                             global_work_size, NULL, 0, NULL, &event);
  assert(CL_err == CL_SUCCESS);
  clWaitForEvents(1, &event);
  assert(clReleaseEvent(event) == CL_SUCCESS);
  clEnqueueReadBuffer(this->command_queue, this->cl_output, CL_TRUE, 0,
                      this->last_nv12_frame.size(),
                      (void *)this->last_nv12_frame.data(), 0, nullptr,
                      nullptr);
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
  kj::ArrayPtr<uint8_t> data((unsigned char *)this->last_nv12_frame.data(),
                             this->last_nv12_frame.size());
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
