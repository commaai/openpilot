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
  inline const char *getData() { return msg->getData(); }
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

#define SEGEMENT_BUF_SIZE 256
class MessageBuilder : public capnp::MessageBuilder {
 public:
  MessageBuilder() : returnedFirstSegment(false), nextSize(2048), stackSegment{} {};

  ~MessageBuilder() {
    for (auto ptr : moreSegments) {
      free(ptr);
    }
  }

  kj::ArrayPtr<capnp::word> allocateSegment(uint minimumSize) {
    if (!returnedFirstSegment) {
      returnedFirstSegment = true;
      uint size = kj::max(minimumSize, SEGEMENT_BUF_SIZE - 1);
      if (size < SEGEMENT_BUF_SIZE) {
        return kj::ArrayPtr<capnp::word>(stackSegment + 1, size);
      }
    }
    uint size = kj::max(minimumSize, nextSize);
    capnp::word *result = (capnp::word *)calloc(size, sizeof(capnp::word));
    moreSegments.add(result);
    nextSize += size;
    return kj::ArrayPtr<capnp::word>(result, size);
  }

  kj::ArrayPtr<capnp::word> toFlatArrayPtr() {
    kj::ArrayPtr<const kj::ArrayPtr<const capnp::word>> segments = getSegmentsForOutput();
    if (segments.size() == 1 && segments[0].begin() == stackSegment + 1) {
      uint32_t *table = (uint32_t *)stackSegment;
      table[0] = 0;
      table[1] = segments[0].size();
      return kj::ArrayPtr<capnp::word>(stackSegment, segments[0].size() + 1);
    } else {
      const size_t size = capnp::computeSerializedSizeInWords(*this);
      flatArray = capnp::messageToFlatArray(segments);
      return flatArray.asPtr();
    }
  }

  int sendTo(PubSocket *sock) {
    auto words = toFlatArrayPtr();
    auto bytes = words.asBytes();
    return sock->send((char *)(bytes.begin()), bytes.size());
  }

 protected:
  bool returnedFirstSegment;
  uint nextSize;
  capnp::word stackSegment[SEGEMENT_BUF_SIZE];
  kj::Array<capnp::word> flatArray;
  kj::Vector<capnp::word *> moreSegments;
};
