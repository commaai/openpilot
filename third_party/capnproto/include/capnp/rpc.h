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

#include <capnp/capability.h>
#include "rpc-prelude.h"

CAPNP_BEGIN_HEADER

namespace kj { class AutoCloseFd; }

namespace capnp {

template <typename VatId, typename ProvisionId, typename RecipientId,
          typename ThirdPartyCapId, typename JoinResult>
class VatNetwork;
template <typename SturdyRefObjectId>
class SturdyRefRestorer;

class MessageReader;

template <typename VatId>
class BootstrapFactory: public _::BootstrapFactoryBase {
  // Interface that constructs per-client bootstrap interfaces. Use this if you want each client
  // who connects to see a different bootstrap interface based on their (authenticated) VatId.
  // This allows an application to bootstrap off of the authentication performed at the VatNetwork
  // level. (Typically VatId is some sort of public key.)
  //
  // This is only useful for multi-party networks. For TwoPartyVatNetwork, there's no reason to
  // use a BootstrapFactory; just specify a single bootstrap capability in this case.

public:
  virtual Capability::Client createFor(typename VatId::Reader clientId) = 0;
  // Create a bootstrap capability appropriate for exposing to the given client. VatNetwork will
  // have authenticated the client VatId before this is called.

private:
  Capability::Client baseCreateFor(AnyStruct::Reader clientId) override;
};

template <typename VatId>
class RpcSystem: public _::RpcSystemBase {
  // Represents the RPC system, which is the portal to objects available on the network.
  //
  // The RPC implementation sits on top of an implementation of `VatNetwork`.  The `VatNetwork`
  // determines how to form connections between vats -- specifically, two-way, private, reliable,
  // sequenced datagram connections.  The RPC implementation determines how to use such connections
  // to manage object references and make method calls.
  //
  // See `makeRpcServer()` and `makeRpcClient()` below for convenient syntax for setting up an
  // `RpcSystem` given a `VatNetwork`.
  //
  // See `ez-rpc.h` for an even simpler interface for setting up RPC in a typical two-party
  // client/server scenario.

public:
  template <typename ProvisionId, typename RecipientId,
            typename ThirdPartyCapId, typename JoinResult>
  RpcSystem(
      VatNetwork<VatId, ProvisionId, RecipientId, ThirdPartyCapId, JoinResult>& network,
      kj::Maybe<Capability::Client> bootstrapInterface);

  template <typename ProvisionId, typename RecipientId,
            typename ThirdPartyCapId, typename JoinResult>
  RpcSystem(
      VatNetwork<VatId, ProvisionId, RecipientId, ThirdPartyCapId, JoinResult>& network,
      BootstrapFactory<VatId>& bootstrapFactory);

  template <typename ProvisionId, typename RecipientId,
            typename ThirdPartyCapId, typename JoinResult,
            typename LocalSturdyRefObjectId>
  RpcSystem(
      VatNetwork<VatId, ProvisionId, RecipientId, ThirdPartyCapId, JoinResult>& network,
      SturdyRefRestorer<LocalSturdyRefObjectId>& restorer);

  RpcSystem(RpcSystem&& other) = default;

  Capability::Client bootstrap(typename VatId::Reader vatId);
  // Connect to the given vat and return its bootstrap interface.

  Capability::Client restore(typename VatId::Reader hostId, AnyPointer::Reader objectId)
      CAPNP_DEPRECATED("Please transition to using a bootstrap interface instead.");
  // ** DEPRECATED **
  //
  // Restores the given SturdyRef from the network and return the capability representing it.
  //
  // `hostId` identifies the host from which to request the ref, in the format specified by the
  // `VatNetwork` in use.  `objectId` is the object ID in whatever format is expected by said host.
  //
  // This method will be removed in a future version of Cap'n Proto. Instead, please transition
  // to using bootstrap(), which is equivalent to calling restore() with a null `objectId`.
  // You may emulate the old concept of object IDs by exporting a bootstrap interface which has
  // methods that can be used to obtain other capabilities by ID.

  void setFlowLimit(size_t words);
  // Sets the incoming call flow limit. If more than `words` worth of call messages have not yet
  // received responses, the RpcSystem will not read further messages from the stream. This can be
  // used as a crude way to prevent a resource exhaustion attack (or bug) in which a peer makes an
  // excessive number of simultaneous calls that consume the receiver's RAM.
  //
  // There are some caveats. When over the flow limit, all messages are blocked, including returns.
  // If the outstanding calls are themselves waiting on calls going in the opposite direction, the
  // flow limit may prevent those calls from completing, leading to deadlock. However, a
  // sufficiently high limit should make this unlikely.
  //
  // Note that a call's parameter size counts against the flow limit until the call returns, even
  // if the recipient calls releaseParams() to free the parameter memory early. This is because
  // releaseParams() may simply indicate that the parameters have been forwarded to another
  // machine, but are still in-memory there. For illustration, say that Alice made a call to Bob
  // who forwarded the call to Carol. Bob has imposed a flow limit on Alice. Alice's calls are
  // being forwarded to Carol, so Bob never keeps the parameters in-memory for more than a brief
  // period. However, the flow limit counts all calls that haven't returned, even if Bob has
  // already freed the memory they consumed. You might argue that the right solution here is
  // instead for Carol to impose her own flow limit on Bob. This has a serious problem, though:
  // Bob might be forwarding requests to Carol on behalf of many different parties, not just Alice.
  // If Alice can pump enough data to hit the Bob -> Carol flow limit, then those other parties
  // will be disrupted. Thus, we can only really impose the limit on the Alice -> Bob link, which
  // only affects Alice. We need that one flow limit to limit Alice's impact on the whole system,
  // so it has to count all in-flight calls.
  //
  // In Sandstorm, flow limits are imposed by the supervisor on calls coming out of a grain, in
  // order to prevent a grain from inundating the system with in-flight calls. In practice, the
  // main time this happens is when a grain is pushing a large file download and doesn't implement
  // proper cooperative flow control.

  // void setTraceEncoder(kj::Function<kj::String(const kj::Exception&)> func);
  //
  // (Inherited from _::RpcSystemBase)
  //
  // Set a function to call to encode exception stack traces for transmission to remote parties.
  // By default, traces are not transmitted at all. If a callback is provided, then the returned
  // string will be sent with the exception. If the remote end is KJ/C++ based, then this trace
  // text ends up being accessible as kj::Exception::getRemoteTrace().
  //
  // Stack traces can sometimes contain sensitive information, so you should think carefully about
  // what information you are willing to reveal to the remote party.

  kj::Promise<void> run() { return RpcSystemBase::run(); }
  // Listens for incoming RPC connections and handles them. Never returns normally, but could throw
  // an exception if the system becomes unable to accept new connections (e.g. because the
  // underlying listen socket becomes broken somehow).
  //
  // For historical reasons, the RpcSystem will actually run itself even if you do not call this.
  // However, if an exception is thrown, the RpcSystem will log the exception to the console and
  // then cease accepting new connections. In this case, your server may be in a broken state, but
  // without restarting. All servers should therefore call run() and handle failures in some way.
};

template <typename VatId, typename ProvisionId, typename RecipientId,
          typename ThirdPartyCapId, typename JoinResult>
RpcSystem<VatId> makeRpcServer(
    VatNetwork<VatId, ProvisionId, RecipientId, ThirdPartyCapId, JoinResult>& network,
    Capability::Client bootstrapInterface);
// Make an RPC server.  Typical usage (e.g. in a main() function):
//
//    MyEventLoop eventLoop;
//    kj::WaitScope waitScope(eventLoop);
//    MyNetwork network;
//    MyMainInterface::Client bootstrap = makeMain();
//    auto server = makeRpcServer(network, bootstrap);
//    kj::NEVER_DONE.wait(waitScope);  // run forever
//
// See also ez-rpc.h, which has simpler instructions for the common case of a two-party
// client-server RPC connection.

template <typename VatId, typename ProvisionId, typename RecipientId,
          typename ThirdPartyCapId, typename JoinResult>
RpcSystem<VatId> makeRpcServer(
    VatNetwork<VatId, ProvisionId, RecipientId, ThirdPartyCapId, JoinResult>& network,
    BootstrapFactory<VatId>& bootstrapFactory);
// Make an RPC server that can serve different bootstrap interfaces to different clients via a
// BootstrapInterface.

template <typename VatId, typename LocalSturdyRefObjectId,
          typename ProvisionId, typename RecipientId, typename ThirdPartyCapId, typename JoinResult>
RpcSystem<VatId> makeRpcServer(
    VatNetwork<VatId, ProvisionId, RecipientId, ThirdPartyCapId, JoinResult>& network,
    SturdyRefRestorer<LocalSturdyRefObjectId>& restorer)
    CAPNP_DEPRECATED("Please transition to using a bootstrap interface instead.");
// ** DEPRECATED **
//
// Create an RPC server which exports multiple main interfaces by object ID. The `restorer` object
// can be used to look up objects by ID.
//
// Please transition to exporting only one interface, which is known as the "bootstrap" interface.
// For backwards-compatibility with old clients, continue to implement SturdyRefRestorer, but
// return the new bootstrap interface when the request object ID is null. When new clients connect
// and request the bootstrap interface, they will get that interface. Eventually, once all clients
// are updated to request only the bootstrap interface, stop implementing SturdyRefRestorer and
// switch to passing the bootstrap capability itself as the second parameter to `makeRpcServer()`.

template <typename VatId, typename ProvisionId,
          typename RecipientId, typename ThirdPartyCapId, typename JoinResult>
RpcSystem<VatId> makeRpcClient(
    VatNetwork<VatId, ProvisionId, RecipientId, ThirdPartyCapId, JoinResult>& network);
// Make an RPC client.  Typical usage (e.g. in a main() function):
//
//    MyEventLoop eventLoop;
//    kj::WaitScope waitScope(eventLoop);
//    MyNetwork network;
//    auto client = makeRpcClient(network);
//    MyCapability::Client cap = client.restore(hostId, objId).castAs<MyCapability>();
//    auto response = cap.fooRequest().send().wait(waitScope);
//    handleMyResponse(response);
//
// See also ez-rpc.h, which has simpler instructions for the common case of a two-party
// client-server RPC connection.

template <typename SturdyRefObjectId>
class SturdyRefRestorer: public _::SturdyRefRestorerBase {
  // ** DEPRECATED **
  //
  // In Cap'n Proto 0.4.x, applications could export multiple main interfaces identified by
  // object IDs. The callback used to map object IDs to objects was `SturdyRefRestorer`, as we
  // imagined this would eventually be used for restoring SturdyRefs as well. In practice, it was
  // never used for real SturdyRefs, only for exporting singleton objects under well-known names.
  //
  // The new preferred strategy is to export only a _single_ such interface, called the
  // "bootstrap interface". That interface can itself have methods for obtaining other objects, of
  // course, but that is up to the app. `SturdyRefRestorer` exists for backwards-compatibility.
  //
  // Hint:  Use SturdyRefRestorer<capnp::Text> to define a server that exports services under
  //   string names.

public:
  virtual Capability::Client restore(typename SturdyRefObjectId::Reader ref) CAPNP_DEPRECATED(
      "Please transition to using bootstrap interfaces instead of SturdyRefRestorer.") = 0;
  // Restore the given object, returning a capability representing it.

private:
  Capability::Client baseRestore(AnyPointer::Reader ref) override final;
};

// =======================================================================================
// VatNetwork

class OutgoingRpcMessage {
  // A message to be sent by a `VatNetwork`.

public:
  virtual AnyPointer::Builder getBody() = 0;
  // Get the message body, which the caller may fill in any way it wants.  (The standard RPC
  // implementation initializes it as a Message as defined in rpc.capnp.)

  virtual void setFds(kj::Array<int> fds) {}
  // Set the list of file descriptors to send along with this message, if FD passing is supported.
  // An implementation may ignore this.

  virtual void send() = 0;
  // Send the message, or at least put it in a queue to be sent later.  Note that the builder
  // returned by `getBody()` remains valid at least until the `OutgoingRpcMessage` is destroyed.

  virtual size_t sizeInWords() = 0;
  // Get the total size of the message, for flow control purposes. Although the caller could
  // also call getBody().targetSize(), doing that would walk the message tree, whereas typical
  // implementations can compute the size more cheaply by summing segment sizes.
};

class IncomingRpcMessage {
  // A message received from a `VatNetwork`.

public:
  virtual AnyPointer::Reader getBody() = 0;
  // Get the message body, to be interpreted by the caller.  (The standard RPC implementation
  // interprets it as a Message as defined in rpc.capnp.)

  virtual kj::ArrayPtr<kj::AutoCloseFd> getAttachedFds() { return nullptr; }
  // If the transport supports attached file descriptors and some were attached to this message,
  // returns them. Otherwise returns an empty array. It is intended that the caller will move the
  // FDs out of this table when they are consumed, possibly leaving behind a null slot. Callers
  // should be careful to check if an FD was already consumed by comparing the slot with `nullptr`.
  // (We don't use Maybe here because moving from a Maybe doesn't make it null, so it would only
  // add confusion. Moving from an AutoCloseFd does in fact make it null.)

  virtual size_t sizeInWords() = 0;
  // Get the total size of the message, for flow control purposes. Although the caller could
  // also call getBody().targetSize(), doing that would walk the message tree, whereas typical
  // implementations can compute the size more cheaply by summing segment sizes.

  static bool isShortLivedRpcMessage(AnyPointer::Reader body);
  // Helper function which computes whether the standard RpcSystem implementation would consider
  // the given message body to be short-lived, meaning it will be dropped before the next message
  // is read. This is useful to implement BufferedMessageStream::IsShortLivedCallback.

  static kj::Function<bool(MessageReader&)> getShortLivedCallback();
  // Returns a function that wraps isShortLivedRpcMessage(). The returned function type matches
  // `BufferedMessageStream::IsShortLivedCallback` (defined in serialize-async.h), but we don't
  // include that header here.
};

class RpcFlowController {
  // Tracks a particular RPC stream in order to implement a flow control algorithm.

public:
  virtual kj::Promise<void> send(kj::Own<OutgoingRpcMessage> message, kj::Promise<void> ack) = 0;
  // Like calling message->send(), but the promise resolves when it's a good time to send the
  // next message.
  //
  // `ack` is a promise that resolves when the message has been acknowledged from the other side.
  // In practice, `message` is typically a `Call` message and `ack` is a `Return`. Note that this
  // means `ack` counts not only time to transmit the message but also time for the remote
  // application to process the message. The flow controller is expected to apply backpressure if
  // the remote application responds slowly. If `ack` rejects, then all outstanding and future
  // sends will propagate the exception.
  //
  // Note that messages sent with this method must still be delivered in the same order as if they
  // had been sent with `message->send()`; they cannot be delayed until later. This is important
  // because the message may introduce state changes in the RPC system that later messages rely on,
  // such as introducing a new Question ID that a later message may reference. Thus, the controller
  // can only create backpressure by having the returned promise resolve slowly.
  //
  // Dropping the returned promise does not cancel the send. Once send() is called, there's no way
  // to stop it.

  virtual kj::Promise<void> waitAllAcked() = 0;
  // Wait for all `ack`s previously passed to send() to finish. It is an error to call send() again
  // after this.

  // ---------------------------------------------------------------------------
  // Common implementations.

  static kj::Own<RpcFlowController> newFixedWindowController(size_t windowSize);
  // Constructs a flow controller that implements a strict fixed window of the given size. In other
  // words, the controller will throttle the stream when the total bytes in-flight exceeds the
  // window.

  class WindowGetter {
  public:
    virtual size_t getWindow() = 0;
  };

  static kj::Own<RpcFlowController> newVariableWindowController(WindowGetter& getter);
  // Like newFixedWindowController(), but the window size is allowed to vary over time. Useful if
  // you have a technique for estimating one good window size for the connection as a whole but not
  // for individual streams. Keep in mind, though, that in situations where the other end of the
  // connection is merely proxying capabilities from a variety of final destinations across a
  // variety of networks, no single window will be appropriate for all streams.

  static constexpr size_t DEFAULT_WINDOW_SIZE = 65536;
  // The window size used by the default implementation of Connection::newStream().
};

template <typename VatId, typename ProvisionId, typename RecipientId,
          typename ThirdPartyCapId, typename JoinResult>
class VatNetwork: public _::VatNetworkBase {
  // Cap'n Proto RPC operates between vats, where a "vat" is some sort of host of objects.
  // Typically one Cap'n Proto process (in the Unix sense) is one vat.  The RPC system is what
  // allows calls between objects hosted in different vats.
  //
  // The RPC implementation sits on top of an implementation of `VatNetwork`.  The `VatNetwork`
  // determines how to form connections between vats -- specifically, two-way, private, reliable,
  // sequenced datagram connections.  The RPC implementation determines how to use such connections
  // to manage object references and make method calls.
  //
  // The most common implementation of VatNetwork is TwoPartyVatNetwork (rpc-twoparty.h).  Most
  // simple client-server apps will want to use it.  (You may even want to use the EZ RPC
  // interfaces in `ez-rpc.h` and avoid all of this.)
  //
  // TODO(someday):  Provide a standard implementation for the public internet.

public:
  class Connection;

  struct ConnectionAndProvisionId {
    // Result of connecting to a vat introduced by another vat.

    kj::Own<Connection> connection;
    // Connection to the new vat.

    kj::Own<OutgoingRpcMessage> firstMessage;
    // An already-allocated `OutgoingRpcMessage` associated with `connection`.  The RPC system will
    // construct this as an `Accept` message and send it.

    Orphan<ProvisionId> provisionId;
    // A `ProvisionId` already allocated inside `firstMessage`, which the RPC system will use to
    // build the `Accept` message.
  };

  class Connection: public _::VatNetworkBase::Connection {
    // A two-way RPC connection.
    //
    // This object may represent a connection that doesn't exist yet, but is expected to exist
    // in the future.  In this case, sent messages will automatically be queued and sent once the
    // connection is ready, so that the caller doesn't need to know the difference.

  public:
    virtual kj::Own<RpcFlowController> newStream() override
        { return RpcFlowController::newFixedWindowController(65536); }
    // Construct a flow controller for a new stream on this connection. The controller can be
    // passed into OutgoingRpcMessage::sendStreaming().
    //
    // The default implementation returns a dummy stream controller that just applies a fixed
    // window of 64k to everything. This always works but may constrain throughput on networks
    // where the bandwidth-delay product is high, while conversely providing too much buffer when
    // the bandwidth-delay product is low.
    //
    // WARNING: The RPC system may keep the `RpcFlowController` object alive past the lifetime of
    //   the `Connection` itself. However, it will not call `send()` any more after the
    //   `Connection` is destroyed.
    //
    // TODO(perf): We should introduce a flow controller implementation that uses a clock to
    //   measure RTT and bandwidth and dynamically update the window size, like BBR.

    // Level 0 features ----------------------------------------------

    virtual typename VatId::Reader getPeerVatId() = 0;
    // Returns the connected vat's authenticated VatId. It is the VatNetwork's responsibility to
    // authenticate this, so that the caller can be assured that they are really talking to the
    // identified vat and not an imposter.

    virtual kj::Own<OutgoingRpcMessage> newOutgoingMessage(uint firstSegmentWordSize) override = 0;
    // Allocate a new message to be sent on this connection.
    //
    // If `firstSegmentWordSize` is non-zero, it should be treated as a hint suggesting how large
    // to make the first segment.  This is entirely a hint and the connection may adjust it up or
    // down.  If it is zero, the connection should choose the size itself.
    //
    // WARNING: The RPC system may keep the `OutgoingRpcMessage` object alive past the lifetime of
    //   the `Connection` itself. However, it will not call `send()` any more after the
    //   `Connection` is destroyed.

    virtual kj::Promise<kj::Maybe<kj::Own<IncomingRpcMessage>>> receiveIncomingMessage() override = 0;
    // Wait for a message to be received and return it.  If the read stream cleanly terminates,
    // return null.  If any other problem occurs, throw an exception.
    //
    // WARNING: The RPC system may keep the `IncomingRpcMessage` object alive past the lifetime of
    //   the `Connection` itself.

    virtual kj::Promise<void> shutdown() override KJ_WARN_UNUSED_RESULT = 0;
    // Waits until all outgoing messages have been sent, then shuts down the outgoing stream. The
    // returned promise resolves after shutdown is complete.

  private:
    AnyStruct::Reader baseGetPeerVatId() override;
  };

  // Level 0 features ------------------------------------------------

  virtual kj::Maybe<kj::Own<Connection>> connect(typename VatId::Reader hostId) = 0;
  // Connect to a VatId.  Note that this method immediately returns a `Connection`, even
  // if the network connection has not yet been established.  Messages can be queued to this
  // connection and will be delivered once it is open.  The caller must attempt to read from the
  // connection to verify that it actually succeeded; the read will fail if the connection
  // couldn't be opened.  Some network implementations may actually start sending messages before
  // hearing back from the server at all, to avoid a round trip.
  //
  // Returns nullptr if `hostId` refers to the local host.

  virtual kj::Promise<kj::Own<Connection>> accept() = 0;
  // Wait for the next incoming connection and return it.

  // Level 4 features ------------------------------------------------
  // TODO(someday)

private:
  kj::Maybe<kj::Own<_::VatNetworkBase::Connection>>
      baseConnect(AnyStruct::Reader hostId) override final;
  kj::Promise<kj::Own<_::VatNetworkBase::Connection>> baseAccept() override final;
};

// =======================================================================================
// ***************************************************************************************
// Inline implementation details start here
// ***************************************************************************************
// =======================================================================================

template <typename VatId>
Capability::Client BootstrapFactory<VatId>::baseCreateFor(AnyStruct::Reader clientId) {
  return createFor(clientId.as<VatId>());
}

template <typename SturdyRef, typename ProvisionId, typename RecipientId,
          typename ThirdPartyCapId, typename JoinResult>
kj::Maybe<kj::Own<_::VatNetworkBase::Connection>>
    VatNetwork<SturdyRef, ProvisionId, RecipientId, ThirdPartyCapId, JoinResult>::
    baseConnect(AnyStruct::Reader ref) {
  auto maybe = connect(ref.as<SturdyRef>());
  return maybe.map([](kj::Own<Connection>& conn) -> kj::Own<_::VatNetworkBase::Connection> {
    return kj::mv(conn);
  });
}

template <typename SturdyRef, typename ProvisionId, typename RecipientId,
          typename ThirdPartyCapId, typename JoinResult>
kj::Promise<kj::Own<_::VatNetworkBase::Connection>>
    VatNetwork<SturdyRef, ProvisionId, RecipientId, ThirdPartyCapId, JoinResult>::baseAccept() {
  return accept().then(
      [](kj::Own<Connection>&& connection) -> kj::Own<_::VatNetworkBase::Connection> {
    return kj::mv(connection);
  });
}

template <typename SturdyRef, typename ProvisionId, typename RecipientId,
          typename ThirdPartyCapId, typename JoinResult>
AnyStruct::Reader VatNetwork<
    SturdyRef, ProvisionId, RecipientId, ThirdPartyCapId, JoinResult>::
    Connection::baseGetPeerVatId() {
  return getPeerVatId();
}

template <typename SturdyRef>
Capability::Client SturdyRefRestorer<SturdyRef>::baseRestore(AnyPointer::Reader ref) {
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
  return restore(ref.getAs<SturdyRef>());
#pragma GCC diagnostic pop
}

template <typename VatId>
template <typename ProvisionId, typename RecipientId,
          typename ThirdPartyCapId, typename JoinResult>
RpcSystem<VatId>::RpcSystem(
      VatNetwork<VatId, ProvisionId, RecipientId, ThirdPartyCapId, JoinResult>& network,
      kj::Maybe<Capability::Client> bootstrap)
    : _::RpcSystemBase(network, kj::mv(bootstrap)) {}

template <typename VatId>
template <typename ProvisionId, typename RecipientId,
          typename ThirdPartyCapId, typename JoinResult>
RpcSystem<VatId>::RpcSystem(
      VatNetwork<VatId, ProvisionId, RecipientId, ThirdPartyCapId, JoinResult>& network,
      BootstrapFactory<VatId>& bootstrapFactory)
    : _::RpcSystemBase(network, bootstrapFactory) {}

template <typename VatId>
template <typename ProvisionId, typename RecipientId,
          typename ThirdPartyCapId, typename JoinResult,
          typename LocalSturdyRefObjectId>
RpcSystem<VatId>::RpcSystem(
      VatNetwork<VatId, ProvisionId, RecipientId, ThirdPartyCapId, JoinResult>& network,
      SturdyRefRestorer<LocalSturdyRefObjectId>& restorer)
    : _::RpcSystemBase(network, restorer) {}

template <typename VatId>
Capability::Client RpcSystem<VatId>::bootstrap(typename VatId::Reader vatId) {
  return baseBootstrap(_::PointerHelpers<VatId>::getInternalReader(vatId));
}

template <typename VatId>
Capability::Client RpcSystem<VatId>::restore(
    typename VatId::Reader hostId, AnyPointer::Reader objectId) {
  return baseRestore(_::PointerHelpers<VatId>::getInternalReader(hostId), objectId);
}

template <typename VatId>
inline void RpcSystem<VatId>::setFlowLimit(size_t words) {
  baseSetFlowLimit(words);
}

template <typename VatId, typename ProvisionId, typename RecipientId,
          typename ThirdPartyCapId, typename JoinResult>
RpcSystem<VatId> makeRpcServer(
    VatNetwork<VatId, ProvisionId, RecipientId, ThirdPartyCapId, JoinResult>& network,
    Capability::Client bootstrapInterface) {
  return RpcSystem<VatId>(network, kj::mv(bootstrapInterface));
}

template <typename VatId, typename ProvisionId, typename RecipientId,
          typename ThirdPartyCapId, typename JoinResult>
RpcSystem<VatId> makeRpcServer(
    VatNetwork<VatId, ProvisionId, RecipientId, ThirdPartyCapId, JoinResult>& network,
    BootstrapFactory<VatId>& bootstrapFactory) {
  return RpcSystem<VatId>(network, bootstrapFactory);
}

template <typename VatId, typename LocalSturdyRefObjectId,
          typename ProvisionId, typename RecipientId, typename ThirdPartyCapId, typename JoinResult>
RpcSystem<VatId> makeRpcServer(
    VatNetwork<VatId, ProvisionId, RecipientId, ThirdPartyCapId, JoinResult>& network,
    SturdyRefRestorer<LocalSturdyRefObjectId>& restorer) {
  return RpcSystem<VatId>(network, restorer);
}

template <typename VatId, typename ProvisionId,
          typename RecipientId, typename ThirdPartyCapId, typename JoinResult>
RpcSystem<VatId> makeRpcClient(
    VatNetwork<VatId, ProvisionId, RecipientId, ThirdPartyCapId, JoinResult>& network) {
  return RpcSystem<VatId>(network, nullptr);
}

}  // namespace capnp

CAPNP_END_HEADER
