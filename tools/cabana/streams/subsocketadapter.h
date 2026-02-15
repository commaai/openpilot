#pragma once

#include <memory>

#include "cereal/messaging/messaging.h"
#include "cereal/messaging/bridge_zmq.h"

class SubSocketAdapter {
public:
  virtual ~SubSocketAdapter() = default;
  virtual Message *receive(bool non_blocking = false) = 0;
};

class MsgqSubSocketAdapter : public SubSocketAdapter {
public:
  MsgqSubSocketAdapter(std::string endpoint,
                       std::string address,
                       bool conflate,
                       bool check_endpoint,
                       size_t segment_size)
    : ctx(Context::create())
  {
    socket.reset(SubSocket::create(ctx.get(), std::move(endpoint), std::move(address),conflate, check_endpoint, segment_size));
  }

  Message *receive(bool non_blocking) override {
    return socket->receive(non_blocking);
  }

private:
  std::unique_ptr<Context> ctx;
  std::unique_ptr<SubSocket> socket;
};

class ZmqSubSocketAdapter : public SubSocketAdapter {
public:
  ZmqSubSocketAdapter(std::string endpoint,
                      std::string address,
                      bool conflate,
                      bool check_endpoint)
  {
    socket.connect(&ctx, std::move(endpoint), std::move(address), conflate, check_endpoint);
  }

  Message *receive(bool non_blocking) override {
    return socket.receive(non_blocking);
  }

private:
  BridgeZmqContext ctx;
  BridgeZmqSubSocket socket;
};
