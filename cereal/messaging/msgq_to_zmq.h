#pragma once

#include <condition_variable>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#define private public
#include "msgq/impl_msgq.h"
#include "msgq/impl_zmq.h"

class MsgqToZmq {
public:
  MsgqToZmq() {}
  void run(const std::vector<std::string> &endpoints, const std::string &ip);

protected:
  void registerSockets();
  void zmqMonitorThread();

  struct SocketPair {
    std::string endpoint;
    std::unique_ptr<ZMQPubSocket> pub_sock;
    std::unique_ptr<MSGQSubSocket> sub_sock;
    int connected_clients = 0;
  };

  std::unique_ptr<MSGQContext> msgq_context;
  std::unique_ptr<ZMQContext> zmq_context;
  std::mutex mutex;
  std::condition_variable cv;
  std::unique_ptr<MSGQPoller> msgq_poller;
  std::map<SubSocket *, ZMQPubSocket *> sub2pub;
  std::vector<SocketPair> socket_pairs;
};
