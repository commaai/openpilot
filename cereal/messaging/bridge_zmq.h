#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include <zmq.h>

#include "msgq/ipc.h"

class BridgeZmqContext {
public:
  BridgeZmqContext();
  void *getRawContext() { return context; }
  ~BridgeZmqContext();

private:
  void *context = nullptr;
};

class BridgeZmqMessage : public Message {
public:
  void init(size_t size);
  void init(char *data, size_t size);
  void close();
  size_t getSize() { return size; }
  char *getData() { return data; }
  ~BridgeZmqMessage();

private:
  char *data = nullptr;
  size_t size = 0;
};

class BridgeZmqSubSocket {
public:
  int connect(BridgeZmqContext *context, std::string endpoint, std::string address, bool conflate = false, bool check_endpoint = true);
  void setTimeout(int timeout);
  Message *receive(bool non_blocking = false);
  void *getRawSocket() { return sock; }
  ~BridgeZmqSubSocket();

private:
  void *sock = nullptr;
  std::string full_endpoint;
};

class BridgeZmqPubSocket {
public:
  int connect(BridgeZmqContext *context, std::string endpoint, bool check_endpoint = true);
  int sendMessage(Message *message);
  int send(char *data, size_t size);
  void *getRawSocket() { return sock; }
  ~BridgeZmqPubSocket();

private:
  void *sock = nullptr;
  std::string full_endpoint;
  int pid = -1;
};

class BridgeZmqPoller {
public:
  void registerSocket(BridgeZmqSubSocket *socket);
  std::vector<BridgeZmqSubSocket *> poll(int timeout);

private:
  static constexpr size_t MAX_BRIDGE_ZMQ_POLLERS = 128;
  std::vector<BridgeZmqSubSocket *> sockets;
  zmq_pollitem_t polls[MAX_BRIDGE_ZMQ_POLLERS] = {};
  size_t num_polls = 0;
};
