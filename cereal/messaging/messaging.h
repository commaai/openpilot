#pragma once

#include <cstddef>
#include <map>
#include <string>
#include <vector>
#include <utility>

#include <capnp/serialize.h>

#include "cereal/gen/cpp/log.capnp.h"
#include "common/timing.h"
#include "msgq/ipc.h"

class SubMaster {
public:
  SubMaster(const std::vector<const char *> &service_list, const std::vector<const char *> &poll = {},
            const char *address = nullptr, const std::vector<const char *> &ignore_alive = {});
  void update(int timeout = 1000);
  void update_msgs(uint64_t current_time, const std::vector<std::pair<std::string, cereal::Event::Reader>> &messages);
  inline bool allAlive(const std::vector<const char *> &service_list = {}) { return all_(service_list, false, true); }
  inline bool allValid(const std::vector<const char *> &service_list = {}) { return all_(service_list, true, false); }
  inline bool allAliveAndValid(const std::vector<const char *> &service_list = {}) { return all_(service_list, true, true); }
  void drain();
  ~SubMaster();

  uint64_t frame = 0;
  bool updated(const char *name) const;
  bool alive(const char *name) const;
  bool valid(const char *name) const;
  uint64_t rcv_frame(const char *name) const;
  uint64_t rcv_time(const char *name) const;
  cereal::Event::Reader &operator[](const char *name) const;

private:
  bool all_(const std::vector<const char *> &service_list, bool valid, bool alive);
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
    event.setLogMonoTime(nanos_since_boot());
    event.setValid(valid);
    return event;
  }

  kj::ArrayPtr<capnp::byte> toBytes() {
    heapArray_ = capnp::messageToFlatArray(*this);
    return heapArray_.asBytes();
  }

  size_t getSerializedSize() {
    return capnp::computeSerializedSizeInWords(*this) * sizeof(capnp::word);
  }

  int serializeToBuffer(unsigned char *buffer, size_t buffer_size) {
    size_t serialized_size = getSerializedSize();
    if (serialized_size > buffer_size) { return -1; }
    kj::ArrayOutputStream out(kj::ArrayPtr<capnp::byte>(buffer, buffer_size));
    capnp::writeMessage(out, *this);
    return serialized_size;
  }

private:
  kj::Array<capnp::word> heapArray_;
};

class PubMaster {
public:
  PubMaster(const std::vector<const char *> &service_list);
  inline int send(const char *name, capnp::byte *data, size_t size) { return sockets_.at(name)->send((char *)data, size); }
  int send(const char *name, MessageBuilder &msg);
  ~PubMaster();

private:
  std::map<std::string, PubSocket *> sockets_;
};

class AlignedBuffer {
public:
  kj::ArrayPtr<const capnp::word> align(const char *data, const size_t size) {
    words_size = size / sizeof(capnp::word) + 1;
    if (aligned_buf.size() < words_size) {
      aligned_buf = kj::heapArray<capnp::word>(words_size < 512 ? 512 : words_size);
    }
    memcpy(aligned_buf.begin(), data, size);
    return aligned_buf.slice(0, words_size);
  }
  inline kj::ArrayPtr<const capnp::word> align(Message *m) {
    return align(m->getData(), m->getSize());
  }
private:
  kj::Array<capnp::word> aligned_buf;
  size_t words_size;
};
