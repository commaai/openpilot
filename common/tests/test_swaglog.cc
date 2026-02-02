#include <fcntl.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

#include <cstring>
#include <iostream>

#include "catch2/catch.hpp"
#include "common/swaglog.h"
#include "common/util.h"
#include "common/version.h"
#include "system/hardware/hw.h"
#include "third_party/json11/json11.hpp"

std::string daemon_name = "testy";
std::string dongle_id = "test_dongle_id";
int LINE_NO = 0;

void log_thread(int thread_id, int msg_cnt) {
  for (int i = 0; i < msg_cnt; ++i) {
    LOGD("%d", thread_id);
    LINE_NO = __LINE__ - 1;
    usleep(1);
  }
}

void recv_log(int thread_cnt, int thread_msg_cnt) {
  // Create Unix datagram socket
  int sock_fd = socket(AF_UNIX, SOCK_DGRAM, 0);
  REQUIRE(sock_fd >= 0);

  // Set non-blocking
  int flags = fcntl(sock_fd, F_GETFL, 0);
  fcntl(sock_fd, F_SETFL, flags | O_NONBLOCK);

  // Get socket path from ipc:// URL
  std::string ipc_path = Path::swaglog_ipc();
  if (ipc_path.rfind("ipc://", 0) == 0) {
    ipc_path = ipc_path.substr(6);
  }

  // Remove existing socket file if present
  unlink(ipc_path.c_str());

  // Bind to socket
  struct sockaddr_un addr;
  memset(&addr, 0, sizeof(addr));
  addr.sun_family = AF_UNIX;
  strncpy(addr.sun_path, ipc_path.c_str(), sizeof(addr.sun_path) - 1);
  REQUIRE(bind(sock_fd, (struct sockaddr*)&addr, sizeof(addr)) == 0);

  std::vector<int> thread_msgs(thread_cnt);
  int total_count = 0;

  for (auto start = std::chrono::steady_clock::now(), now = start;
       now < start + std::chrono::seconds{1} && total_count < (thread_cnt * thread_msg_cnt);
       now = std::chrono::steady_clock::now()) {
    char buf[4096] = {};
    ssize_t len = recv(sock_fd, buf, sizeof(buf), 0);
    if (len <= 0) {
      if (errno == EAGAIN || errno == EINTR || errno == EWOULDBLOCK) continue;
      break;
    }

    REQUIRE(buf[0] == CLOUDLOG_DEBUG);
    std::string err;
    auto msg = json11::Json::parse(buf + 1, err);
    REQUIRE(!msg.is_null());

    REQUIRE(msg["levelnum"].int_value() == CLOUDLOG_DEBUG);
    REQUIRE_THAT(msg["filename"].string_value(), Catch::Contains("test_swaglog.cc"));
    REQUIRE(msg["funcname"].string_value() == "log_thread");
    REQUIRE(msg["lineno"].int_value() == LINE_NO);

    auto ctx = msg["ctx"];

    REQUIRE(ctx["daemon"].string_value() == daemon_name);
    REQUIRE(ctx["dongle_id"].string_value() == dongle_id);
    REQUIRE(ctx["dirty"].bool_value() == true);

    REQUIRE(ctx["version"].string_value() == COMMA_VERSION);

    std::string device = Hardware::get_name();
    REQUIRE(ctx["device"].string_value() == device);

    int thread_id = atoi(msg["msg"].string_value().c_str());
    REQUIRE((thread_id >= 0 && thread_id < thread_cnt));
    thread_msgs[thread_id]++;
    total_count++;
  }
  for (int i = 0; i < thread_cnt; ++i) {
    INFO("thread :" << i);
    REQUIRE(thread_msgs[i] == thread_msg_cnt);
  }
  close(sock_fd);
  unlink(ipc_path.c_str());
}

TEST_CASE("swaglog") {
  setenv("MANAGER_DAEMON", daemon_name.c_str(), 1);
  setenv("DONGLE_ID", dongle_id.c_str(), 1);
  setenv("dirty", "1", 1);
  const int thread_cnt = 5;
  const int thread_msg_cnt = 100;

  std::vector<std::thread> log_threads;
  for (int i = 0; i < thread_cnt; ++i) {
    log_threads.push_back(std::thread(log_thread, i, thread_msg_cnt));
  }
  for (auto &t : log_threads) t.join();

  recv_log(thread_cnt, thread_msg_cnt);
}
