#pragma once
#include <cstddef>
#include <map>
#include <string>
#include <vector>
#include <capnp/serialize.h>
#include "../gen/cpp/log.capnp.h"

#ifdef __APPLE__
#define CLOCK_BOOTTIME CLOCK_MONOTONIC
#endif

#define MSG_MULTIPLE_PUBLISHERS 100

bool messaging_use_zmq();

class Context {
public:
  virtual void * getRawContext() = 0;
  static Context * create();
  virtual ~Context(){};
};

class Message {
public:
  virtual void init(size_t size) = 0;
  virtual void init(char * data, size_t size) = 0;
  virtual void close() = 0;
  virtual size_t getSize() = 0;
  virtual char * getData() = 0;
  virtual ~Message(){};
};


class SubSocket {
public:
  virtual int connect(Context *context, std::string endpoint, std::string address, bool conflate=false, bool check_endpoint=true) = 0;
  virtual void setTimeout(int timeout) = 0;
  virtual Message *receive(bool non_blocking=false) = 0;
  virtual void * getRawSocket() = 0;
  static SubSocket * create();
  static SubSocket * create(Context * context, std::string endpoint, std::string address="127.0.0.1", bool conflate=false, bool check_endpoint=true);
  virtual ~SubSocket(){};
};

class PubSocket {
public:
  virtual int connect(Context *context, std::string endpoint, bool check_endpoint=true) = 0;
  virtual int sendMessage(Message *message) = 0;
  virtual int send(char *data, size_t size) = 0;
  static PubSocket * create();
  static PubSocket * create(Context * context, std::string endpoint, bool check_endpoint=true);
  static PubSocket * create(Context * context, std::string endpoint, int port, bool check_endpoint=true);
  virtual ~PubSocket(){};
};

class Poller {
public:
  virtual void registerSocket(SubSocket *socket) = 0;
  virtual std::vector<SubSocket*> poll(int timeout) = 0;
  static Poller * create();
  static Poller * create(std::vector<SubSocket*> sockets);
  virtual ~Poller(){};
};

class SubMaster {
public:
  SubMaster(const std::initializer_list<const char *> &service_list,
            const char *address = nullptr, const std::initializer_list<const char *> &ignore_alive = {});
  int update(int timeout = 1000);
  inline bool allAlive(const std::initializer_list<const char *> &service_list = {}) { return all_(service_list, false, true); }
  inline bool allValid(const std::initializer_list<const char *> &service_list = {}) { return all_(service_list, true, false); }
  inline bool allAliveAndValid(const std::initializer_list<const char *> &service_list = {}) { return all_(service_list, true, true); }
  void drain();
  ~SubMaster();

  uint64_t frame = 0;
  bool updated(const char *name) const;
  uint64_t rcv_frame(const char *name) const;
  cereal::Event::Reader &operator[](const char *name);

private:
  bool all_(const std::initializer_list<const char *> &service_list, bool valid, bool alive);
  Poller *poller_ = nullptr;
  struct SubMessage;
  std::map<SubSocket *, SubMessage *> messages_;
  std::map<std::string, SubMessage *> services_;
};

class MessageBuilder : public capnp::MallocMessageBuilder {
public:
  MessageBuilder() = default;

  cereal::Event::Builder initEvent(bool valid = true) {
    cereal::Event::Builder event = initRoot<cereal::Event>();
    struct timespec t;
    clock_gettime(CLOCK_BOOTTIME, &t);
    uint64_t current_time = t.tv_sec * 1000000000ULL + t.tv_nsec;
    event.setLogMonoTime(current_time);
    event.setValid(valid);
    return event;
  }

  kj::ArrayPtr<capnp::byte> toBytes() {
    heapArray_ = capnp::messageToFlatArray(*this);
    return heapArray_.asBytes();
  }

private:
  kj::Array<capnp::word> heapArray_;
};

class PubMaster {
public:
  PubMaster(const std::initializer_list<const char *> &service_list);
  inline int send(const char *name, capnp::byte *data, size_t size) { return sockets_.at(name)->send((char *)data, size); }
  int send(const char *name, MessageBuilder &msg);
  ~PubMaster();

private:
  std::map<std::string, PubSocket *> sockets_;
};
