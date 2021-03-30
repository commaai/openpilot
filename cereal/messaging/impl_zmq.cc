#include <cassert>
#include <cstring>
#include <iostream>
#include <cstdlib>
#include <cerrno>

#include <zmq.h>

#include "services.h"
#include "impl_zmq.hpp"

static int get_port(std::string endpoint) {
  int port = -1;
  for (const auto& it : services) {
    std::string name = it.name;
    if (name == endpoint) {
      port = it.port;
      break;
    }
  }

  assert(port >= 0);
  return port;
}

ZMQContext::ZMQContext() {
  context = zmq_ctx_new();
}

ZMQContext::~ZMQContext() {
  zmq_ctx_term(context);
}

void ZMQMessage::init(size_t sz) {
  size = sz;
  data = new char[size];
}

void ZMQMessage::init(char * d, size_t sz) {
  size = sz;
  data = new char[size];
  memcpy(data, d, size);
}

void ZMQMessage::close() {
  if (size > 0){
    delete[] data;
  }
  size = 0;
}

ZMQMessage::~ZMQMessage() {
  this->close();
}


int ZMQSubSocket::connect(Context *context, std::string endpoint, std::string address, bool conflate, bool check_endpoint){
  sock = zmq_socket(context->getRawContext(), ZMQ_SUB);
  if (sock == NULL){
    return -1;
  }

  zmq_setsockopt(sock, ZMQ_SUBSCRIBE, "", 0);

  if (conflate){
    int arg = 1;
    zmq_setsockopt(sock, ZMQ_CONFLATE, &arg, sizeof(int));
  }

  int reconnect_ivl = 500;
  zmq_setsockopt(sock, ZMQ_RECONNECT_IVL_MAX, &reconnect_ivl, sizeof(reconnect_ivl));

  full_endpoint = "tcp://" + address + ":";
  if (check_endpoint){
    full_endpoint += std::to_string(get_port(endpoint));
  } else {
    full_endpoint += endpoint;
  }

  return zmq_connect(sock, full_endpoint.c_str());
}


Message * ZMQSubSocket::receive(bool non_blocking){
  zmq_msg_t msg;
  assert(zmq_msg_init(&msg) == 0);

  int flags = non_blocking ? ZMQ_DONTWAIT : 0;
  int rc = zmq_msg_recv(&msg, sock, flags);
  Message *r = NULL;

  if (rc >= 0){
    // Make a copy to ensure the data is aligned
    r = new ZMQMessage;
    r->init((char*)zmq_msg_data(&msg), zmq_msg_size(&msg));
  }

  zmq_msg_close(&msg);
  return r;
}

void ZMQSubSocket::setTimeout(int timeout){
  zmq_setsockopt(sock, ZMQ_RCVTIMEO, &timeout, sizeof(int));
}

ZMQSubSocket::~ZMQSubSocket(){
  zmq_close(sock);
}

int ZMQPubSocket::connect(Context *context, std::string endpoint, bool check_endpoint){
  sock = zmq_socket(context->getRawContext(), ZMQ_PUB);
  if (sock == NULL){
    return -1;
  }

  full_endpoint = "tcp://*:";
  if (check_endpoint){
    full_endpoint += std::to_string(get_port(endpoint));
  } else {
    full_endpoint += endpoint;
  }

  return zmq_bind(sock, full_endpoint.c_str());
}

int ZMQPubSocket::sendMessage(Message *message){
  return zmq_send(sock, message->getData(), message->getSize(), ZMQ_DONTWAIT);
}

int ZMQPubSocket::send(char *data, size_t size){
  return zmq_send(sock, data, size, ZMQ_DONTWAIT);
}

ZMQPubSocket::~ZMQPubSocket(){
  zmq_close(sock);
}


void ZMQPoller::registerSocket(SubSocket * socket){
  assert(num_polls + 1 < MAX_POLLERS);
  polls[num_polls].socket = socket->getRawSocket();
  polls[num_polls].events = ZMQ_POLLIN;

  sockets.push_back(socket);
  num_polls++;
}

std::vector<SubSocket*> ZMQPoller::poll(int timeout){
  std::vector<SubSocket*> r;

  int rc = zmq_poll(polls, num_polls, timeout);
  if (rc < 0){
    return r;
  }

  for (size_t i = 0; i < num_polls; i++){
    if (polls[i].revents){
      r.push_back(sockets[i]);
    }
  }

  return r;
}
