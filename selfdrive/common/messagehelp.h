#pragma once
#include <capnp/serialize.h>

#include "cereal/gen/cpp/log.capnp.h"
#include "messaging.hpp"
class MessageReader {
 public:
  MessageReader(Message *msg) : msg(msg), buf(nullptr), msg_reader(nullptr) {}
  ~MessageReader() {
    if (msg_reader) delete msg_reader;
    if (buf) free(buf);
    if (msg) delete msg;
  }
  inline operator bool() { return msg != NULL; }
  inline const char* getData() { return msg->getData(); }
  inline size_t getSize() { return msg->getSize(); }
  cereal::Event::Reader &getEvent() {
    if (!msg_reader) {
      msg_reader = newReader();
      event = msg_reader->getRoot<cereal::Event>();
    }
    return event;
  }

 private:
  capnp::FlatArrayMessageReader *newReader() {
    const char *data = msg->getData();
    const size_t size = msg->getSize();
    if (((reinterpret_cast<uintptr_t>(data)) % sizeof(capnp::word) == 0) && size % sizeof(capnp::word) == 0) {
      return new capnp::FlatArrayMessageReader(kj::ArrayPtr<capnp::word>((capnp::word *)data, size / sizeof(capnp::word)));
    } else {
      const size_t words = size / sizeof(capnp::word) + 1;
      buf = (capnp::word *)malloc(words * sizeof(capnp::word));
      memcpy(buf, data, size);
      return new capnp::FlatArrayMessageReader(kj::ArrayPtr<capnp::word>(buf, words));
    }
  }
  capnp::word *buf;
  Message *msg;
  capnp::FlatArrayMessageReader *msg_reader;
  cereal::Event::Reader event;
};
