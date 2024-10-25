#include "cereal/messaging/msgq_to_zmq.h"

#include <cassert>

#include "common/util.h"

extern ExitHandler do_exit;

// Max messages to process per socket per poll
constexpr int MAX_MESSAGES_PER_SOCKET = 50;

static std::string recv_zmq_msg(void *sock) {
  zmq_msg_t msg;
  zmq_msg_init(&msg);
  std::string ret;
  if (zmq_msg_recv(&msg, sock, 0) > 0) {
    ret.assign((char *)zmq_msg_data(&msg), zmq_msg_size(&msg));
  }
  zmq_msg_close(&msg);
  return ret;
}

void MsgqToZmq::run(const std::vector<std::string> &endpoints, const std::string &ip) {
  zmq_context = std::make_unique<ZMQContext>();
  msgq_context = std::make_unique<MSGQContext>();

  // Create ZMQPubSockets for each endpoint
  for (const auto &endpoint : endpoints) {
    auto &socket_pair = socket_pairs.emplace_back();
    socket_pair.endpoint = endpoint;
    socket_pair.pub_sock = std::make_unique<ZMQPubSocket>();
    int ret = socket_pair.pub_sock->connect(zmq_context.get(), endpoint);
    if (ret != 0) {
      printf("Failed to create ZMQ publisher for [%s]: %s\n", endpoint.c_str(), zmq_strerror(zmq_errno()));
      return;
    }
  }

  // Start ZMQ monitoring thread to monitor socket events
  std::thread thread(&MsgqToZmq::zmqMonitorThread, this);

  // Main loop for processing messages
  while (!do_exit) {
    {
      std::unique_lock lk(mutex);
      cv.wait(lk, [this]() { return do_exit || !sub2pub.empty(); });
      if (do_exit) break;

      for (auto sub_sock : msgq_poller->poll(100)) {
        // Process messages for each socket
        ZMQPubSocket *pub_sock = sub2pub.at(sub_sock);
        for (int i = 0; i < MAX_MESSAGES_PER_SOCKET; ++i) {
          auto msg = std::unique_ptr<Message>(sub_sock->receive(true));
          if (!msg) break;

          while (pub_sock->sendMessage(msg.get()) == -1) {
            if (errno != EINTR) break;
          }
        }
      }
    }
    util::sleep_for(1);  // Give zmqMonitorThread a chance to acquire the mutex
  }

  thread.join();
}

void MsgqToZmq::zmqMonitorThread() {
  std::vector<zmq_pollitem_t> pollitems;

  // Set up ZMQ monitor for each pub socket
  for (int i = 0; i < socket_pairs.size(); ++i) {
    std::string addr = "inproc://op-bridge-monitor-" + std::to_string(i);
    zmq_socket_monitor(socket_pairs[i].pub_sock->sock, addr.c_str(), ZMQ_EVENT_ACCEPTED | ZMQ_EVENT_DISCONNECTED);

    void *monitor_socket = zmq_socket(zmq_context->getRawContext(), ZMQ_PAIR);
    zmq_connect(monitor_socket, addr.c_str());
    pollitems.emplace_back(zmq_pollitem_t{.socket = monitor_socket, .events = ZMQ_POLLIN});
  }

  while (!do_exit) {
    int ret = zmq_poll(pollitems.data(), pollitems.size(), 1000);
    if (ret < 0) {
      if (errno == EINTR) {
        // Due to frequent EINTR signals from msgq, introduce a brief delay (200 ms)
        // to reduce CPU usage during retry attempts.
        util::sleep_for(200);
      }
      continue;
    }

    for (int i = 0; i < pollitems.size(); ++i) {
      if (pollitems[i].revents & ZMQ_POLLIN) {
        // First frame in message contains event number and value
        std::string frame = recv_zmq_msg(pollitems[i].socket);
        if (frame.empty()) continue;

        uint16_t event_type = *(uint16_t *)(frame.data());

        // Second frame in message contains event address
        frame = recv_zmq_msg(pollitems[i].socket);
        if (frame.empty()) continue;

        std::unique_lock lk(mutex);
        auto &pair = socket_pairs[i];
        if (event_type & ZMQ_EVENT_ACCEPTED) {
          printf("socket [%s] connected\n", pair.endpoint.c_str());
          if (++pair.connected_clients == 1) {
            // Create new MSGQ subscriber socket and map to ZMQ publisher
            pair.sub_sock = std::make_unique<MSGQSubSocket>();
            pair.sub_sock->connect(msgq_context.get(), pair.endpoint, "127.0.0.1");
            sub2pub[pair.sub_sock.get()] = pair.pub_sock.get();
            registerSockets();
          }
        } else if (event_type & ZMQ_EVENT_DISCONNECTED) {
          printf("socket [%s] disconnected\n", pair.endpoint.c_str());
          if (pair.connected_clients == 0 || --pair.connected_clients == 0) {
            // Remove MSGQ subscriber socket from mapping and reset it
            sub2pub.erase(pair.sub_sock.get());
            pair.sub_sock.reset(nullptr);
            registerSockets();
          }
        }
        cv.notify_one();
      }
    }
  }

  // Clean up monitor sockets
  for (int i = 0; i < pollitems.size(); ++i) {
    zmq_socket_monitor(socket_pairs[i].pub_sock->sock, nullptr, 0);
    zmq_close(pollitems[i].socket);
  }
  cv.notify_one();
}

void MsgqToZmq::registerSockets() {
  msgq_poller = std::make_unique<MSGQPoller>();
  for (const auto &socket_pair : socket_pairs) {
    if (socket_pair.sub_sock) {
      msgq_poller->registerSocket(socket_pair.sub_sock.get());
    }
  }
}
