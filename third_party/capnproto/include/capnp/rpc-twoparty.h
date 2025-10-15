// Copyright (c) 2013-2014 Sandstorm Development Group, Inc. and contributors
// Licensed under the MIT License:
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#pragma once

#include "rpc.h"
#include <capnp/message.h>
#include <kj/async-io.h>
#include <capnp/serialize-async.h>
#include <capnp/rpc-twoparty.capnp.h>
#include <kj/one-of.h>

CAPNP_BEGIN_HEADER

namespace capnp {

namespace rpc {
  namespace twoparty {
    typedef VatId SturdyRefHostId;  // For backwards-compatibility with version 0.4.
  }
}

typedef VatNetwork<rpc::twoparty::VatId, rpc::twoparty::ProvisionId,
    rpc::twoparty::RecipientId, rpc::twoparty::ThirdPartyCapId, rpc::twoparty::JoinResult>
    TwoPartyVatNetworkBase;

class TwoPartyVatNetwork: public TwoPartyVatNetworkBase,
                          private TwoPartyVatNetworkBase::Connection,
                          private RpcFlowController::WindowGetter {
  // A `VatNetwork` that consists of exactly two parties communicating over an arbitrary byte
  // stream.  This is used to implement the common case of a client/server network.
  //
  // See `ez-rpc.h` for a simple interface for setting up two-party clients and servers.
  // Use `TwoPartyVatNetwork` only if you need the advanced features.

public:
  TwoPartyVatNetwork(MessageStream& msgStream,
                     rpc::twoparty::Side side, ReaderOptions receiveOptions = ReaderOptions(),
                     const kj::MonotonicClock& clock = kj::systemCoarseMonotonicClock());
  TwoPartyVatNetwork(MessageStream& msgStream, uint maxFdsPerMessage,
                     rpc::twoparty::Side side, ReaderOptions receiveOptions = ReaderOptions(),
                     const kj::MonotonicClock& clock = kj::systemCoarseMonotonicClock());
  TwoPartyVatNetwork(kj::AsyncIoStream& stream, rpc::twoparty::Side side,
                     ReaderOptions receiveOptions = ReaderOptions(),
                     const kj::MonotonicClock& clock = kj::systemCoarseMonotonicClock());
  TwoPartyVatNetwork(kj::AsyncCapabilityStream& stream, uint maxFdsPerMessage,
                     rpc::twoparty::Side side, ReaderOptions receiveOptions = ReaderOptions(),
                     const kj::MonotonicClock& clock = kj::systemCoarseMonotonicClock());
  // To support FD passing, pass an AsyncCapabilityStream or a MessageStream which supports
  // fd passing, and `maxFdsPerMessage`, which specifies the maximum number of file descriptors
  // to accept from the peer in any one RPC message. It is important to keep maxFdsPerMessage
  // low in order to stop DoS attacks that fill up your FD table.
  //
  // Note that this limit applies only to incoming messages; outgoing messages are allowed to have
  // more FDs. Sometimes it makes sense to enforce a limit of zero in one direction while having
  // a non-zero limit in the other. For example, in a supervisor/sandbox scenario, typically there
  // are many use cases for passing FDs from supervisor to sandbox but no use case for vice versa.
  // The supervisor may be configured not to accept any FDs from the sandbox in order to reduce
  // risk of DoS attacks.
  //
  // clock is used for calculating the oldest queued message age, which is a useful metric for
  // detecting queue overload

  ~TwoPartyVatNetwork() noexcept(false);
  KJ_DISALLOW_COPY_AND_MOVE(TwoPartyVatNetwork);

  kj::Promise<void> onDisconnect() { return disconnectPromise.addBranch(); }
  // Returns a promise that resolves when the peer disconnects.

  rpc::twoparty::Side getSide() { return side; }

  size_t getCurrentQueueSize() { return currentQueueSize; }
  // Get the number of bytes worth of outgoing messages that are currently queued in memory waiting
  // to be sent on this connection. This may be useful for backpressure.

  size_t getCurrentQueueCount() { return queuedMessages.size(); }
  // Get the count of outgoing messages that are currently queued in memory waiting
  // to be sent on this connection. This may be useful for backpressure.

  kj::Duration getOutgoingMessageWaitTime();
  // Get how long the current outgoing message has been waiting to be sent on this connection.
  // Returns 0 if the queue is empty. This may be useful for backpressure.

  // implements VatNetwork -----------------------------------------------------

  kj::Maybe<kj::Own<TwoPartyVatNetworkBase::Connection>> connect(
      rpc::twoparty::VatId::Reader ref) override;
  kj::Promise<kj::Own<TwoPartyVatNetworkBase::Connection>> accept() override;

private:
  class OutgoingMessageImpl;
  class IncomingMessageImpl;

  kj::OneOf<MessageStream*, kj::Own<MessageStream>> stream;
  // The underlying stream, which we may or may not own. Get a reference to
  // this with getStream, rather than reading it directly.

  uint maxFdsPerMessage;
  rpc::twoparty::Side side;
  MallocMessageBuilder peerVatId;
  ReaderOptions receiveOptions;
  bool accepted = false;

  bool solSndbufUnimplemented = false;
  // Whether stream.getsockopt(SO_SNDBUF) has been observed to throw UNIMPLEMENTED.

  kj::Canceler readCanceler;
  kj::Maybe<kj::Exception> readCancelReason;
  // Used to propagate write errors into (permanent) read errors.

  kj::Maybe<kj::Promise<void>> previousWrite;
  // Resolves when the previous write completes.  This effectively serves as the write queue.
  // Becomes null when shutdown() is called.

  kj::Own<kj::PromiseFulfiller<kj::Own<TwoPartyVatNetworkBase::Connection>>> acceptFulfiller;
  // Fulfiller for the promise returned by acceptConnectionAsRefHost() on the client side, or the
  // second call on the server side.  Never fulfilled, because there is only one connection.

  kj::ForkedPromise<void> disconnectPromise = nullptr;

  kj::Vector<kj::Own<OutgoingMessageImpl>> queuedMessages;
  size_t currentQueueSize = 0;
  const kj::MonotonicClock& clock;
  kj::TimePoint currentOutgoingMessageSendTime;

  class FulfillerDisposer: public kj::Disposer {
    // Hack:  TwoPartyVatNetwork is both a VatNetwork and a VatNetwork::Connection.  When the RPC
    //   system detects (or initiates) a disconnection, it drops its reference to the Connection.
    //   When all references have been dropped, then we want disconnectPromise to be fulfilled.
    //   So we hand out Own<Connection>s with this disposer attached, so that we can detect when
    //   they are dropped.

  public:
    mutable kj::Own<kj::PromiseFulfiller<void>> fulfiller;
    mutable uint refcount = 0;

    void disposeImpl(void* pointer) const override;
  };
  FulfillerDisposer disconnectFulfiller;


  TwoPartyVatNetwork(
      kj::OneOf<MessageStream*, kj::Own<MessageStream>>&& stream,
      uint maxFdsPerMessage,
      rpc::twoparty::Side side,
      ReaderOptions receiveOptions,
      const kj::MonotonicClock& clock);

  MessageStream& getStream();

  kj::Own<TwoPartyVatNetworkBase::Connection> asConnection();
  // Returns a pointer to this with the disposer set to disconnectFulfiller.

  // implements Connection -----------------------------------------------------

  kj::Own<RpcFlowController> newStream() override;
  rpc::twoparty::VatId::Reader getPeerVatId() override;
  kj::Own<OutgoingRpcMessage> newOutgoingMessage(uint firstSegmentWordSize) override;
  kj::Promise<kj::Maybe<kj::Own<IncomingRpcMessage>>> receiveIncomingMessage() override;
  kj::Promise<void> shutdown() override;

  // implements WindowGetter ---------------------------------------------------

  size_t getWindow() override;
};

class TwoPartyServer: private kj::TaskSet::ErrorHandler {
  // Convenience class which implements a simple server which accepts connections on a listener
  // socket and services them as two-party connections.

public:
  explicit TwoPartyServer(Capability::Client bootstrapInterface,
      kj::Maybe<kj::Function<kj::String(const kj::Exception&)>> traceEncoder = nullptr);
  // `traceEncoder`, if provided, will be passed on to `rpcSystem.setTraceEncoder()`.

  void accept(kj::Own<kj::AsyncIoStream>&& connection);
  void accept(kj::Own<kj::AsyncCapabilityStream>&& connection, uint maxFdsPerMessage);
  // Accepts the connection for servicing.

  kj::Promise<void> accept(kj::AsyncIoStream& connection) KJ_WARN_UNUSED_RESULT;
  kj::Promise<void> accept(kj::AsyncCapabilityStream& connection, uint maxFdsPerMessage)
      KJ_WARN_UNUSED_RESULT;
  // Accept connection without taking ownership. The returned promise resolves when the client
  // disconnects. Dropping the promise forcefully cancels the RPC protocol.
  //
  // You probably can't do anything with `connection` after the RPC protocol has terminated, other
  // than to close it. The main reason to use these methods rather than the ownership-taking ones
  // is if your stream object becomes invalid outside some scope, so you want to make sure to
  // cancel all usage of it before that by cancelling the promise.

  kj::Promise<void> listen(kj::ConnectionReceiver& listener);
  // Listens for connections on the given listener. The returned promise never resolves unless an
  // exception is thrown while trying to accept. You may discard the returned promise to cancel
  // listening.

  kj::Promise<void> listenCapStreamReceiver(
      kj::ConnectionReceiver& listener, uint maxFdsPerMessage);
  // Listen with support for FD transfers. `listener.accept()` must return instances of
  // AsyncCapabilityStream, otherwise this will crash.

  kj::Promise<void> drain() { return tasks.onEmpty(); }
  // Resolves when all clients have disconnected.
  //
  // Only considers clients whose connections TwoPartyServer took ownership of.

private:
  Capability::Client bootstrapInterface;
  kj::Maybe<kj::Function<kj::String(const kj::Exception&)>> traceEncoder;
  kj::TaskSet tasks;

  struct AcceptedConnection;

  void taskFailed(kj::Exception&& exception) override;
};

class TwoPartyClient {
  // Convenience class which implements a simple client.

public:
  explicit TwoPartyClient(kj::AsyncIoStream& connection);
  explicit TwoPartyClient(kj::AsyncCapabilityStream& connection, uint maxFdsPerMessage);
  TwoPartyClient(kj::AsyncIoStream& connection, Capability::Client bootstrapInterface,
                 rpc::twoparty::Side side = rpc::twoparty::Side::CLIENT);
  TwoPartyClient(kj::AsyncCapabilityStream& connection, uint maxFdsPerMessage,
                 Capability::Client bootstrapInterface,
                 rpc::twoparty::Side side = rpc::twoparty::Side::CLIENT);

  Capability::Client bootstrap();
  // Get the server's bootstrap interface.

  inline kj::Promise<void> onDisconnect() { return network.onDisconnect(); }

  void setTraceEncoder(kj::Function<kj::String(const kj::Exception&)> func);
  // Forwarded to rpcSystem.setTraceEncoder().

  size_t getCurrentQueueSize() { return network.getCurrentQueueSize(); }
  size_t getCurrentQueueCount() { return network.getCurrentQueueCount(); }
  kj::Duration getOutgoingMessageWaitTime() { return network.getOutgoingMessageWaitTime(); }

private:
  TwoPartyVatNetwork network;
  RpcSystem<rpc::twoparty::VatId> rpcSystem;
};

}  // namespace capnp

CAPNP_END_HEADER
