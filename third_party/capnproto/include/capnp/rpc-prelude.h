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

// This file contains a bunch of internal declarations that must appear before rpc.h can start.
// We don't define these directly in rpc.h because it makes the file hard to read.

#pragma once

#include <capnp/capability.h>
#include "persistent.capnp.h"

CAPNP_BEGIN_HEADER

namespace capnp {

class OutgoingRpcMessage;
class IncomingRpcMessage;
class RpcFlowController;

template <typename SturdyRefHostId>
class RpcSystem;

namespace _ {  // private

class VatNetworkBase {
  // Non-template version of VatNetwork.  Ignore this class; see VatNetwork in rpc.h.

public:
  class Connection;

  struct ConnectionAndProvisionId {
    kj::Own<Connection> connection;
    kj::Own<OutgoingRpcMessage> firstMessage;
    Orphan<AnyPointer> provisionId;
  };

  class Connection {
  public:
    virtual kj::Own<OutgoingRpcMessage> newOutgoingMessage(uint firstSegmentWordSize) = 0;
    virtual kj::Promise<kj::Maybe<kj::Own<IncomingRpcMessage>>> receiveIncomingMessage() = 0;
    virtual kj::Promise<void> shutdown() = 0;
    virtual AnyStruct::Reader baseGetPeerVatId() = 0;
    virtual kj::Own<RpcFlowController> newStream() = 0;
  };
  virtual kj::Maybe<kj::Own<Connection>> baseConnect(AnyStruct::Reader vatId) = 0;
  virtual kj::Promise<kj::Own<Connection>> baseAccept() = 0;
};

class SturdyRefRestorerBase {
public:
  virtual Capability::Client baseRestore(AnyPointer::Reader ref) = 0;
};

class BootstrapFactoryBase {
  // Non-template version of BootstrapFactory.  Ignore this class; see BootstrapFactory in rpc.h.
public:
  virtual Capability::Client baseCreateFor(AnyStruct::Reader clientId) = 0;
};

class RpcSystemBase {
  // Non-template version of RpcSystem.  Ignore this class; see RpcSystem in rpc.h.

public:
  RpcSystemBase(VatNetworkBase& network, kj::Maybe<Capability::Client> bootstrapInterface);
  RpcSystemBase(VatNetworkBase& network, BootstrapFactoryBase& bootstrapFactory);
  RpcSystemBase(VatNetworkBase& network, SturdyRefRestorerBase& restorer);
  RpcSystemBase(RpcSystemBase&& other) noexcept;
  ~RpcSystemBase() noexcept(false);

  void setTraceEncoder(kj::Function<kj::String(const kj::Exception&)> func);

  kj::Promise<void> run();

private:
  class Impl;
  kj::Own<Impl> impl;

  Capability::Client baseBootstrap(AnyStruct::Reader vatId);
  Capability::Client baseRestore(AnyStruct::Reader vatId, AnyPointer::Reader objectId);
  void baseSetFlowLimit(size_t words);

  template <typename>
  friend class capnp::RpcSystem;
};

}  // namespace _ (private)
}  // namespace capnp

CAPNP_END_HEADER
