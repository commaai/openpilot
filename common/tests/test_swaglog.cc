#include <fcntl.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

#include <atomic>
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
    usleep(100);  // Small delay to avoid overwhelming the socket buffer
  }
}

TEST_CASE("swaglog") {
  setenv("MANAGER_DAEMON", daemon_name.c_str(), 1);
  setenv("DONGLE_ID", dongle_id.c_str(), 1);
  setenv("dirty", "1", 1);
  const int thread_cnt = 5;
  const int thread_msg_cnt = 100;

  // Create and bind receiver socket BEFORE starting senders
  // (datagram sockets drop messages if no receiver is bound)
  int sock_fd = socket(AF_UNIX, SOCK_DGRAM, 0);
  REQUIRE(sock_fd >= 0);

  // Increase receive buffer to handle burst of messages
  int rcvbuf = 1024 * 1024;  // 1MB buffer
  setsockopt(sock_fd, SOL_SOCKET, SO_RCVBUF, &rcvbuf, sizeof(rcvbuf));

  int flags = fcntl(sock_fd, F_GETFL, 0);
  fcntl(sock_fd, F_SETFL, flags | O_NONBLOCK);

  std::string ipc_path = Path::swaglog_ipc();
  if (ipc_path.rfind("ipc://", 0) == 0) {
    ipc_path = ipc_path.substr(6);
  }
  unlink(ipc_path.c_str());

  struct sockaddr_un addr;
  memset(&addr, 0, sizeof(addr));
  addr.sun_family = AF_UNIX;
  strncpy(addr.sun_path, ipc_path.c_str(), sizeof(addr.sun_path) - 1);
  REQUIRE(bind(sock_fd, (struct sockaddr*)&addr, sizeof(addr)) == 0);

  // Shared state for receiver thread
  std::vector<int> thread_msgs(thread_cnt);
  std::atomic<int> total_count{0};
  std::atomic<bool> stop_receiver{false};

  // Start receiver thread BEFORE senders to receive messages concurrently
  std::thread receiver_thread([&]() {
    while (!stop_receiver || total_count < (thread_cnt * thread_msg_cnt)) {
      char buf[4096] = {};
      ssize_t len = recv(sock_fd, buf, sizeof(buf), 0);
      if (len <= 0) {
        if (errno == EAGAIN || errno == EINTR || errno == EWOULDBLOCK) {
          usleep(100);
          continue;
        }
        break;
      }

      if (buf[0] != CLOUDLOG_DEBUG) continue;
      std::string err;
      auto msg = json11::Json::parse(buf + 1, err);
      if (msg.is_null()) continue;

      int thread_id = atoi(msg["msg"].string_value().c_str());
      if (thread_id >= 0 && thread_id < thread_cnt) {
        thread_msgs[thread_id]++;
        total_count++;
      }

      if (total_count >= thread_cnt * thread_msg_cnt) break;
    }
  });

  // Now start senders
  std::vector<std::thread> log_threads;
  for (int i = 0; i < thread_cnt; ++i) {
    log_threads.push_back(std::thread(log_thread, i, thread_msg_cnt));
  }
  for (auto &t : log_threads) t.join();

  // Signal receiver to stop and wait for it
  stop_receiver = true;
  // Give receiver a bit more time to drain any remaining messages
  usleep(100000);  // 100ms
  receiver_thread.join();

  // Verify all messages were received
  for (int i = 0; i < thread_cnt; ++i) {
    INFO("thread :" << i);
    REQUIRE(thread_msgs[i] == thread_msg_cnt);
  }
  close(sock_fd);
  unlink(ipc_path.c_str());
}
