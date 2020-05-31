#pragma once
#include <cstddef>
#include <map>
#include <string>
#include <vector>
#include <capnp/serialize.h>
#include "../gen/cpp/log.capnp.h"

#define MSG_MULTIPLE_PUBLISHERS 100

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
  virtual int connect(Context *context, std::string endpoint, std::string address, bool conflate=false) = 0;
  virtual void setTimeout(int timeout) = 0;
  virtual Message *receive(bool non_blocking=false) = 0;
  virtual void * getRawSocket() = 0;
  static SubSocket * create();
  static SubSocket * create(Context * context, std::string endpoint);
  static SubSocket * create(Context * context, std::string endpoint, std::string address);
  static SubSocket * create(Context * context, std::string endpoint, std::string address, bool conflate);
  virtual ~SubSocket(){};
};

class PubSocket {
public:
  virtual int connect(Context *context, std::string endpoint) = 0;
  virtual int sendMessage(Message *message) = 0;
  virtual int send(char *data, size_t size) = 0;
  static PubSocket * create();
  static PubSocket * create(Context * context, std::string endpoint);
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
  bool updated(const char *name) const;
  void drain();
  cereal::Event::Reader &operator[](const char *name);
  ~SubMaster();

 private:
  bool all_(const std::initializer_list<const char *> &service_list, bool valid, bool alive);
  Poller *poller_ = nullptr;
  uint64_t frame_ = 0;
  struct SubMessage;
  std::map<SubSocket *, SubMessage *> messages_;
  std::map<std::string, SubMessage *> services_;
};

class PubMaster {
 public:
  PubMaster(const std::initializer_list<const char *> &service_list);
  inline int send(const char *name, capnp::byte *data, size_t size) { return sockets_.at(name)->send((char *)data, size); }
  int send(const char *name, capnp::MessageBuilder &msg);
  ~PubMaster();

 private:
  std::map<std::string, PubSocket *> sockets_;
};
