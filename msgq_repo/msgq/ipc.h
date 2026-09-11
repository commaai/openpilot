#pragma once

#include <cstddef>
#include <map>
#include <string>
#include <vector>
#include <utility>
#include <time.h>



#ifdef __APPLE__
#define CLOCK_BOOTTIME CLOCK_MONOTONIC
#endif

#define MSG_MULTIPLE_PUBLISHERS 100

class Context {
public:
  static Context * create();
  ~Context(){}
};

class Message {
private:
  char * data = nullptr;
  size_t size = 0;
public:
  void init(size_t size);
  void init(char * data, size_t size);
  void takeOwnership(char * data, size_t size);
  void close();
  size_t getSize(){return size;}
  char * getData(){return data;}
  ~Message();
};


class SubSocket {
public:
  virtual int connect(Context *context, std::string endpoint, std::string address, bool conflate=false, bool check_endpoint=true, size_t segment_size=0) = 0;
  virtual void setTimeout(int timeout) = 0;
  virtual Message *receive(bool non_blocking=false) = 0;
  static SubSocket * create();
  static SubSocket * create(Context * context, std::string endpoint, std::string address="127.0.0.1", bool conflate=false, bool check_endpoint=true, size_t segment_size=0);
  virtual ~SubSocket(){}
};

class PubSocket {
private:
  struct msgq_queue_t * q = nullptr;
public:
  int connect(Context *context, std::string endpoint, bool check_endpoint=true, size_t segment_size=0);
  int sendMessage(Message *message);
  int send(char *data, size_t size);
  bool all_readers_updated();
  static PubSocket * create();
  static PubSocket * create(Context * context, std::string endpoint, bool check_endpoint=true, size_t segment_size=0);
  ~PubSocket();
};

class Poller {
public:
  virtual void registerSocket(SubSocket *socket) = 0;
  virtual std::vector<SubSocket*> poll(int timeout) = 0;
  static Poller * create();
  static Poller * create(std::vector<SubSocket*> sockets);
  virtual ~Poller(){}
};
