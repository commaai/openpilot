#include "cereal/messaging/bridge_zmq.h"

#include <cassert>
#include <cstring>
#include <unistd.h>

static size_t fnv1a_hash(const std::string &str) {
  const size_t fnv_prime = 0x100000001b3;
  size_t hash_value = 0xcbf29ce484222325;
  for (char c : str) {
    hash_value ^= (unsigned char)c;
    hash_value *= fnv_prime;
  }
  return hash_value;
}

// FIXME: This is a hack to get the port number from the socket name, might have collisions.
static int get_port(std::string endpoint) {
  size_t hash_value = fnv1a_hash(endpoint);
  int start_port = 8023;
  int max_port = 65535;
  return start_port + (hash_value % (max_port - start_port));
}

BridgeZmqContext::BridgeZmqContext() {
  context = zmq_ctx_new();
}

BridgeZmqContext::~BridgeZmqContext() {
  if (context != nullptr) {
    zmq_ctx_term(context);
  }
}

void BridgeZmqMessage::init(size_t sz) {
  size = sz;
  data = new char[size];
}

void BridgeZmqMessage::init(char *d, size_t sz) {
  size = sz;
  data = new char[size];
  memcpy(data, d, size);
}

void BridgeZmqMessage::close() {
  if (size > 0) {
    delete[] data;
  }
  data = nullptr;
  size = 0;
}

BridgeZmqMessage::~BridgeZmqMessage() {
  close();
}

int BridgeZmqSubSocket::connect(BridgeZmqContext *context, std::string endpoint, std::string address, bool conflate, bool check_endpoint) {
  sock = zmq_socket(context->getRawContext(), ZMQ_SUB);
  if (sock == nullptr) {
    return -1;
  }

  zmq_setsockopt(sock, ZMQ_SUBSCRIBE, "", 0);

  if (conflate) {
    int arg = 1;
    zmq_setsockopt(sock, ZMQ_CONFLATE, &arg, sizeof(int));
  }

  int reconnect_ivl = 500;
  zmq_setsockopt(sock, ZMQ_RECONNECT_IVL_MAX, &reconnect_ivl, sizeof(reconnect_ivl));

  full_endpoint = "tcp://" + address + ":";
  if (check_endpoint) {
    full_endpoint += std::to_string(get_port(endpoint));
  } else {
    full_endpoint += endpoint;
  }

  return zmq_connect(sock, full_endpoint.c_str());
}

void BridgeZmqSubSocket::setTimeout(int timeout) {
  zmq_setsockopt(sock, ZMQ_RCVTIMEO, &timeout, sizeof(int));
}

Message *BridgeZmqSubSocket::receive(bool non_blocking) {
  zmq_msg_t msg;
  assert(zmq_msg_init(&msg) == 0);

  int flags = non_blocking ? ZMQ_DONTWAIT : 0;
  int rc = zmq_msg_recv(&msg, sock, flags);

  Message *ret = nullptr;
  if (rc >= 0) {
    ret = new BridgeZmqMessage;
    ret->init((char *)zmq_msg_data(&msg), zmq_msg_size(&msg));
  }

  zmq_msg_close(&msg);
  return ret;
}

BridgeZmqSubSocket::~BridgeZmqSubSocket() {
  if (sock != nullptr) {
    zmq_close(sock);
  }
}

int BridgeZmqPubSocket::connect(BridgeZmqContext *context, std::string endpoint, bool check_endpoint) {
  sock = zmq_socket(context->getRawContext(), ZMQ_PUB);
  if (sock == nullptr) {
    return -1;
  }

  full_endpoint = "tcp://*:";
  if (check_endpoint) {
    full_endpoint += std::to_string(get_port(endpoint));
  } else {
    full_endpoint += endpoint;
  }

  // ZMQ pub sockets cannot be shared between processes, so we need to ensure pid stays the same.
  pid = getpid();

  return zmq_bind(sock, full_endpoint.c_str());
}

int BridgeZmqPubSocket::sendMessage(Message *message) {
  assert(pid == getpid());
  return zmq_send(sock, message->getData(), message->getSize(), ZMQ_DONTWAIT);
}

int BridgeZmqPubSocket::send(char *data, size_t size) {
  assert(pid == getpid());
  return zmq_send(sock, data, size, ZMQ_DONTWAIT);
}

BridgeZmqPubSocket::~BridgeZmqPubSocket() {
  if (sock != nullptr) {
    zmq_close(sock);
  }
}

void BridgeZmqPoller::registerSocket(BridgeZmqSubSocket *socket) {
  assert(num_polls + 1 < (sizeof(polls) / sizeof(polls[0])));
  polls[num_polls].socket = socket->getRawSocket();
  polls[num_polls].events = ZMQ_POLLIN;

  sockets.push_back(socket);
  num_polls++;
}

std::vector<BridgeZmqSubSocket *> BridgeZmqPoller::poll(int timeout) {
  std::vector<BridgeZmqSubSocket *> ret;

  int rc = zmq_poll(polls, num_polls, timeout);
  if (rc < 0) {
    return ret;
  }

  for (size_t i = 0; i < num_polls; i++) {
    if (polls[i].revents) {
      ret.push_back(sockets[i]);
    }
  }

  return ret;
}
