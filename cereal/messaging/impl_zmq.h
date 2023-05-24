#pragma once

#include <zmq.h>
#include <string>
#include <vector>

#include "cereal/messaging/messaging.h"

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
  int connect(Context *context, std::string endpoint, std::string address, bool conflate=false, bool check_endpoint=true);
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
  int connect(Context *context, std::string endpoint, bool check_endpoint=true);
  int sendMessage(Message *message);
  int send(char *data, size_t size);
  bool all_readers_updated();
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
