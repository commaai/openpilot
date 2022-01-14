#include <zmq.h>
#include <iostream>
#define CATCH_CONFIG_MAIN
#include "catch2/catch.hpp"

#include "json11.hpp"
#include "selfdrive/common/swaglog.h"
#include "selfdrive/common/util.h"
#include "selfdrive/common/version.h"
#include "selfdrive/hardware/hw.h"

const char *SWAGLOG_ADDR = "ipc:///tmp/logmessage";
std::string dongle_id = "test_dongle_id";

void log_thread(int msg, int msg_cnt) {
  for (int i = 0; i < msg_cnt; ++i) {
    LOGD("%d", msg);
    usleep(1);
  }
}

void send_stop_msg(void *zctx) {
  void *sock = zmq_socket(zctx, ZMQ_PUSH);
  zmq_connect(sock, SWAGLOG_ADDR);
  zmq_send(sock, "", 0, ZMQ_NOBLOCK);
  zmq_close(sock);
}

void recv_log(void *zctx, int thread_cnt, int thread_msg_cnt) {
  void *sock = zmq_socket(zctx, ZMQ_PULL);
  zmq_bind(sock, SWAGLOG_ADDR);
  std::vector<int> thread_msgs(thread_cnt);

  while (true) {
    char buf[4096] = {};
    if (zmq_recv(sock, buf, sizeof(buf), 0) == 0) break;

    REQUIRE(buf[0] == CLOUDLOG_DEBUG);
    std::string err;
    auto msg = json11::Json::parse(buf + 1, err);
    REQUIRE(!msg.is_null());

    REQUIRE(msg["levelnum"].int_value() == CLOUDLOG_DEBUG);
    REQUIRE_THAT(msg["filename"].string_value(), Catch::Contains("test_swaglog.cc"));
    REQUIRE(msg["funcname"].string_value() == "log_thread");
    REQUIRE(msg["lineno"].int_value() == 17);

    auto ctx = msg["ctx"];
    REQUIRE(ctx["dongle_id"].string_value() == dongle_id);
    REQUIRE(ctx["version"].string_value() == COMMA_VERSION);
    REQUIRE(ctx["dirty"].bool_value() == true);
    std::string device = "pc";
    if (Hardware::EON()) {
      device = "eon";
    } else if (Hardware::TICI()) {
      device = "tici";
    }
    REQUIRE(ctx["device"].string_value() == device);

    int thread_id = atoi(msg["msg"].string_value().c_str());
    REQUIRE((thread_id >= 0 && thread_id < thread_cnt));
    thread_msgs[thread_id]++;
  }
  for (int i = 0; i < thread_cnt; ++i) {
    REQUIRE(thread_msgs[i] == thread_msg_cnt);
  }
  zmq_close(sock);
}

TEST_CASE("swaglog") {
  setenv("DONGLE_ID", dongle_id.c_str(), 1);
  setenv("dirty", "1", 1);
  const int thread_cnt = 5;
  const int thread_msg_cnt = 100;

  void *zctx = zmq_ctx_new();
  send_stop_msg(zctx);
  std::vector<std::thread> log_threads;
  for (int i = 0; i < thread_cnt; ++i) {
    log_threads.push_back(std::thread(log_thread, i, thread_msg_cnt));
  }

  for (auto &t : log_threads) t.join();
  recv_log(zctx, thread_cnt, thread_msg_cnt);
  zmq_ctx_destroy(zctx);
}
