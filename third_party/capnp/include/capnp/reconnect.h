// Copyright (c) 2020 Cloudflare, Inc. and contributors
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
#include <kj/function.h>

CAPNP_BEGIN_HEADER

namespace capnp {

template <typename ConnectFunc>
auto autoReconnect(ConnectFunc&& connect);
// Creates a capability that reconstructs itself every time it becomes disconnected.
//
// `connect()` is a function which is invoked to initially construct the capability, and then
// invoked again each time the capability is found to be disconnected. `connect()` may return
// any capability `Client` type.
//
// Example usage might look like:
//
//     Foo::Client foo = autoReconnect([&rpcSystem, vatId]() {
//       return rpcSystem.bootstrap(vatId).castAs<RootType>().getFooRequest().send().getFoo();
//     });
//
// The given function is initially called synchronously, and the returned `foo` is a wrapper
// around what the function returned. But any time this capability becomes disconnected, the
// function is invoked again, and future calls are directed to the new result.
//
// Any call that is in-flight when the capability becomes disconnected still fails with a
// DISCONNECTED exception. The caller should respond by retrying, as a retry will target the
// newly-reconnected capability. However, the caller should limit the number of times it retries,
// to avoid an infinite loop in the case that the DISCONNECTED exception actually represents a
// permanent problem. Consider using `kj::retryOnDisconnect()` to implement this behavior.

template <typename ConnectFunc>
auto lazyAutoReconnect(ConnectFunc&& connect);
// The same as autoReconnect, but doesn't call the provided connect function until the first
// time the capability is used. Note that only the initial connection is lazy -- upon
// disconnected errors this will still reconnect eagerly.

// =======================================================================================
// inline implementation details

Capability::Client autoReconnect(kj::Function<Capability::Client()> connect);
template <typename ConnectFunc>
auto autoReconnect(ConnectFunc&& connect) {
  return autoReconnect(kj::Function<Capability::Client()>(kj::fwd<ConnectFunc>(connect)))
      .castAs<FromClient<kj::Decay<decltype(connect())>>>();
}

Capability::Client lazyAutoReconnect(kj::Function<Capability::Client()> connect);
template <typename ConnectFunc>
auto lazyAutoReconnect(ConnectFunc&& connect) {
  return lazyAutoReconnect(kj::Function<Capability::Client()>(kj::fwd<ConnectFunc>(connect)))
      .castAs<FromClient<kj::Decay<decltype(connect())>>>();
}

}  // namespace capnp

CAPNP_END_HEADER
