#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>
#include <chrono>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "common/util.h"
#include "msgq/impl_msgq.h"
#include "msgq/impl_zmq.h"
#define protected public  // Access protected members for testing
#include "cereal/messaging/msgq_to_zmq.h"

extern ExitHandler do_exit;

TEST_CASE("MsgqToZmq_MultiEndpoint", "[MsgqToZmq]") {
  MsgqToZmq bridge;
  std::vector<std::string> endpoints = {"can", "sendcan"};
  const std::string zmq_address = "127.0.0.1";
  auto bridge_thread = std::thread([&]() { bridge.run(endpoints, zmq_address); });
  util::sleep_for(50);

  INFO("Initialization and Multi-Endpoint Forwarding");
  {
    // Setup publisher and subscriber sockets for each endpoint
    MSGQContext msgq_context;
    std::unordered_map<std::string, MSGQPubSocket> pub_sockets;
    ZMQContext zmq_context;
    std::unordered_map<std::string, ZMQSubSocket> sub_sockets;

    for (const auto& endpoint : endpoints) {
      pub_sockets[endpoint].connect(&msgq_context, endpoint, true);
      sub_sockets[endpoint].connect(&zmq_context, endpoint, zmq_address);
    }

    INFO("Wait for bridge to establish connections for all endpoints");
    {
      std::unique_lock lk(bridge.mutex);
      bool ret = bridge.cv.wait_for(lk, std::chrono::milliseconds(500),
                                    [&bridge, &endpoints]() {
                                      return bridge.sub2pub.size() == endpoints.size();
                                    });
      REQUIRE(ret == true);
    }
    REQUIRE(bridge.socket_pairs.size() == endpoints.size());
    REQUIRE(bridge.sub2pub.size() == endpoints.size());

    util::sleep_for(50);  // Allow sockets to stabilize

    // Test message forwarding for each endpoint
    const int num_messages = 100;
    std::unordered_map<std::string, std::vector<std::string>> sent_messages;

    // Prepare and send messages for each endpoint
    for (const auto& endpoint : endpoints) {
      auto& messages = sent_messages[endpoint];
      for (int i = 0; i < num_messages; ++i) {
        messages.push_back(endpoint + "_msg" + std::to_string(i));
      }
      for (int i = 0; i < num_messages; ++i) {
        INFO("Sending " << endpoint << " message " << i << ": " << messages[i]);
        pub_sockets[endpoint].send(messages[i].data(), messages[i].size());
      }
    }

    // Receive and verify messages for each endpoint
    std::unordered_map<std::string, std::vector<std::string>> received_messages;
    auto start = std::chrono::steady_clock::now();
    for (const auto& endpoint : endpoints) {
      while (received_messages[endpoint].size() < num_messages) {
        if (std::chrono::steady_clock::now() - start > std::chrono::seconds(3)) {
          FAIL("Timeout waiting for " << num_messages << " messages on " << endpoint
                                      << "; received " << received_messages[endpoint].size());
        }
        auto msg = sub_sockets[endpoint].receive(true);
        if (msg) {
          received_messages[endpoint].emplace_back(msg->getData(), msg->getSize());
          delete msg;
        } else {
          std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
      }
    }

    // Verify order and content for each endpoint
    for (const auto& endpoint : endpoints) {
      const auto& sent = sent_messages[endpoint];
      const auto& received = received_messages[endpoint];
      REQUIRE(received.size() == sent.size());
      for (size_t i = 0; i < sent.size(); ++i) {
        INFO("Checking " << endpoint << " message " << i << ": sent '" << sent[i]
                         << "' vs received '" << received[i] << "'");
        CHECK(received[i] == sent[i]);
      }
    }
  }

  INFO("Verifying bridge cleanup and transition to sleep mode after all subscribers disconnect");
  {
    std::unique_lock lk(bridge.mutex);
    auto start = std::chrono::steady_clock::now();
    while (bridge.sub2pub.size() > 0) {
      if (std::chrono::steady_clock::now() - start > std::chrono::seconds(3)) {
        FAIL("Timeout waiting for sub2pub to be empty; size = " << bridge.sub2pub.size());
      }
      bridge.cv.wait_for(lk, std::chrono::milliseconds(50));
    }
    REQUIRE(bridge.sub2pub.size() == 0);
  }

  // Shutdown the bridge
  do_exit = true;
  bridge_thread.join();
}
