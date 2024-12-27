//
// Example that demonstrates offboard control using attitude, velocity control
// in NED (North-East-Down), and velocity control in body (Forward-Right-Down)
// coordinates.
//

#include <cerrno>
#include <chrono>
#include <cmath>
#include <future>
#include <iostream>
#include <thread>

#include <netinet/in.h>
#include <netinet/ip.h>
#include <sys/socket.h>

#include <mavsdk/mavsdk.h>
#include <mavsdk/plugins/action/action.h>
#include <mavsdk/plugins/offboard/offboard.h>
#include <mavsdk/plugins/telemetry/telemetry.h>

using namespace mavsdk;
using std::chrono::milliseconds;
using std::chrono::seconds;
using std::this_thread::sleep_for;

void usage(const std::string &bin_name) {
  std::cerr
      << "Usage : " << bin_name << " <connection_url>\n"
      << "Connection URL format should be :\n"
      << " For TCP : tcp://[server_host][:server_port]\n"
      << " For UDP : udp://[bind_host][:bind_port]\n"
      << " For Serial : serial:///path/to/serial/dev[:baudrate]\n"
      << "For example, to connect to the simulator use URL: udp://:14540\n";
}

//
// Does Offboard control using body co-ordinates.
// Body coordinates really means world coordinates rotated by the yaw of the
// vehicle, so if the vehicle pitches down, the forward axis does still point
// forward and not down into the ground.
//
// returns true if everything went well in Offboard control.
//
bool offb_ctrl_body(mavsdk::Offboard &offboard) {
  std::cout << "Starting Offboard velocity control in body coordinates\n";

  // Send it once before starting offboard, otherwise it will be rejected.
  Offboard::VelocityBodyYawspeed stay{};
  offboard.set_velocity_body(stay);

  Offboard::Result offboard_result = offboard.start();
  if (offboard_result != Offboard::Result::Success) {
    std::cerr << "Offboard start failed: " << offboard_result << '\n';
    return false;
  }
  std::cout << "Offboard started\n";

  std::cout << "Turn clock-wise and climb\n";
  Offboard::VelocityBodyYawspeed cc_and_climb{};
  cc_and_climb.down_m_s = -1.0f;
  cc_and_climb.yawspeed_deg_s = 60.0f;
  offboard.set_velocity_body(cc_and_climb);
  sleep_for(seconds(5));

  std::cout << "Turn back anti-clockwise\n";
  Offboard::VelocityBodyYawspeed ccw{};
  ccw.down_m_s = -1.0f;
  ccw.yawspeed_deg_s = -60.0f;
  offboard.set_velocity_body(ccw);
  sleep_for(seconds(5));

  std::cout << "Wait for a bit\n";
  offboard.set_velocity_body(stay);
  sleep_for(seconds(2));

  std::cout << "Fly a circle\n";
  Offboard::VelocityBodyYawspeed circle{};
  circle.forward_m_s = 5.0f;
  circle.yawspeed_deg_s = 30.0f;
  offboard.set_velocity_body(circle);
  sleep_for(seconds(15));

  std::cout << "Wait for a bit\n";
  offboard.set_velocity_body(stay);
  sleep_for(seconds(5));

  std::cout << "Fly a circle sideways\n";
  circle.right_m_s = -5.0f;
  circle.yawspeed_deg_s = 30.0f;
  offboard.set_velocity_body(circle);
  sleep_for(seconds(15));

  std::cout << "Wait for a bit\n";
  offboard.set_velocity_body(stay);
  sleep_for(seconds(8));

  offboard_result = offboard.stop();
  if (offboard_result != Offboard::Result::Success) {
    std::cerr << "Offboard stop failed: " << offboard_result << '\n';
    return false;
  }
  std::cout << "Offboard stopped\n";

  return true;
}

int main(int argc, char **argv) {
  if (argc != 2) {
    usage(argv[0]);
    return 1;
  }

  Mavsdk mavsdk{Mavsdk::Configuration{Mavsdk::ComponentType::GroundStation}};
  ConnectionResult connection_result = mavsdk.add_any_connection(argv[1]);

  if (connection_result != ConnectionResult::Success) {
    std::cerr << "Connection failed: " << connection_result << '\n';
    return 1;
  }

  auto system = mavsdk.first_autopilot(3.0);
  if (!system) {
    std::cerr << "Timed out waiting for system\n";
    return 1;
  }

  // Instantiate plugins.
  auto action = Action{system.value()};
  auto offboard = Offboard{system.value()};
  auto telemetry = Telemetry{system.value()};

  while (!telemetry.health_all_ok()) {
    std::cout << "Waiting for system to be ready\n";
    sleep_for(seconds(1));
  }
  std::cout << "System is ready\n";

  const auto arm_result = action.arm();
  if (arm_result != Action::Result::Success) {
    std::cerr << "Arming failed: " << arm_result << '\n';
    return 1;
  }
  std::cout << "Armed\n";

  const auto takeoff_result = action.takeoff();
  if (takeoff_result != Action::Result::Success) {
    std::cerr << "Takeoff failed: " << takeoff_result << '\n';
    return 1;
  }

  auto in_air_promise = std::promise<void>{};
  auto in_air_future = in_air_promise.get_future();
  Telemetry::LandedStateHandle handle = telemetry.subscribe_landed_state(
      [&telemetry, &in_air_promise, &handle](Telemetry::LandedState state) {
        if (state == Telemetry::LandedState::InAir) {
          std::cout << "Taking off has finished\n.";
          telemetry.unsubscribe_landed_state(handle);
          in_air_promise.set_value();
        }
      });
  in_air_future.wait_for(seconds(10));
  if (in_air_future.wait_for(seconds(3)) == std::future_status::timeout) {
    std::cerr << "Takeoff timed out.\n";
    return 1;
  }

  auto tcp_socket = socket(AF_INET, SOCK_STREAM, 0);
  if (tcp_socket < 0) {
    std::cerr << "Failed to create socket: " << strerror(errno) << '\n';
    return errno;
  }
  sockaddr_in server_addr{.sin_family = AF_INET,
                          .sin_port = htons(8096),
                          .sin_addr = {
                              .s_addr = INADDR_ANY,
                          },
                          .sin_zero = {0, 0, 0, 0, 0, 0, 0, 0}};
  if (bind(tcp_socket, reinterpret_cast<sockaddr *>(&server_addr),
           sizeof(server_addr)) < 0) {
    std::cerr << "Failed to bind socket: " << strerror(errno) << '\n';
    return errno;
  };
  if (listen(tcp_socket, 10) < 0) {
    std::cerr << "Failed to listen on socket: " << strerror(errno) << '\n';
    return errno;
  }
  std::flush(std::cout);
  std::cout << "Waiting for connection...\n";
  sockaddr_in client_addr{};
  socklen_t client_addr_len = sizeof(client_addr);
  int client_socket_fd = accept(tcp_socket, reinterpret_cast<sockaddr *>(&client_addr),
                              &client_addr_len);
  if (client_socket_fd < 0) {
    std::cerr << "Failed to accept connection: " << strerror(errno) << '\n';
    return errno;
  }
  std::cout << "Accepted connection from " << client_addr.sin_addr.s_addr << '\n';

  //  using body co-ordinates
  if (!offb_ctrl_body(offboard)) {
    return 1;
  }

  const auto land_result = action.land();
  if (land_result != Action::Result::Success) {
    std::cerr << "Landing failed: " << land_result << '\n';
    return 1;
  }

  // Check if vehicle is still in air
  while (telemetry.in_air()) {
    std::cout << "Vehicle is landing...\n";
    sleep_for(seconds(1));
  }
  std::cout << "Landed!\n";

  // We are relying on auto-disarming but let's keep watching the telemetry for
  // a bit longer.
  sleep_for(seconds(3));
  std::cout << "Finished...\n";

  return 0;
}
