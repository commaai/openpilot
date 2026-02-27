#pragma once

#include <condition_variable>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "msgq/impl_msgq.h"
#include "cereal/messaging/bridge_zmq.h"

class MsgqToZmq {
public:
  MsgqToZmq() {}
  void run(const std::vector<std::string> &endpoints, const std::string &ip);

protected:
  void registerSockets();
  void zmqMonitorThread();

  struct SocketPair {
    std::string endpoint;
    std::unique_ptr<BridgeZmqPubSocket> pub_sock;
    std::unique_ptr<MSGQSubSocket> sub_sock;
    int connected_clients = 0;
  };

  std::unique_ptr<Context> msgq_context;
  std::unique_ptr<BridgeZmqContext> zmq_context;
  std::mutex mutex;
  std::condition_variable cv;
  std::unique_ptr<MSGQPoller> msgq_poller;
  std::map<SubSocket *, BridgeZmqPubSocket *> sub2pub;
  std::vector<SocketPair> socket_pairs;
};
