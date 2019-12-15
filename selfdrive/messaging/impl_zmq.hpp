#pragma once
#include "messaging.hpp"
#include <zmq.h>
#include <string>

#define MAX_POLLERS 128

class ZMQContext : public Context {
private:
  void * context = NULL;
public:
  ZMQContext();
  void * getRawContext() {return context;}
  ~ZMQContext();
};

class ZMQMessage : public Message {
private:
  char * data;
  size_t size;
public:
  void init(size_t size);
  void init(char *data, size_t size);
  size_t getSize(){return size;}
  char * getData(){return data;}
  void close();
  ~ZMQMessage();
};

class ZMQSubSocket : public SubSocket {
private:
  void * sock;
  std::string full_endpoint;
public:
  void connect(Context *context, std::string endpoint, std::string address, bool conflate=false);
  void setTimeout(int timeout);
  void * getRawSocket() {return sock;}
  Message *receive(bool non_blocking=false);
  ~ZMQSubSocket();
};

class ZMQPubSocket : public PubSocket {
private:
  void * sock;
  std::string full_endpoint;
public:
  void connect(Context *context, std::string endpoint);
  int sendMessage(Message *message);
  int send(char *data, size_t size);
  ~ZMQPubSocket();
};

class ZMQPoller : public Poller {
private:
  std::vector<SubSocket*> sockets;
  zmq_pollitem_t polls[MAX_POLLERS];
  size_t num_polls = 0;

public:
  void registerSocket(SubSocket *socket);
  std::vector<SubSocket*> poll(int timeout);
  ~ZMQPoller(){};
};
