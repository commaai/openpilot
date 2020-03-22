// Copyright (c) 2015 Sandstorm Development Group, Inc. and contributors
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

#ifndef CAPNP_MEMBRANE_H_
#define CAPNP_MEMBRANE_H_
// In capability theory, a "membrane" is a wrapper around a capability which (usually) forwards
// calls but recursively wraps capabilities in those calls in the same membrane. The purpose of a
// membrane is to enforce a barrier between two capabilities that cannot be bypassed by merely
// introducing new objects.
//
// The most common use case for a membrane is revocation: Say Alice wants to give Bob a capability
// to access Carol, but wants to be able to revoke this capability later. Alice can accomplish this
// by wrapping Carol in a revokable wrapper which passes through calls until such a time as Alice
// indicates it should be revoked, after which all calls through the wrapper will throw exceptions.
// However, a naive wrapper approach has a problem: if Bob makes a call to Carol and sends a new
// capability in that call, or if Carol returns a capability to Bob in the response to a call, then
// the two are now able to communicate using this new capability, which Alice cannot revoke. In
// order to avoid this problem, Alice must use not just a wrapper but a "membrane", which
// recursively wraps all objects that pass through it in either direction. Thus, all connections
// formed between Bob and Carol (originating from Alice's original introduction) can be revoked
// together by revoking the membrane.
//
// Note that when a capability is passed into a membrane and then passed back out, the result is
// the original capability, not a double-membraned capability. This means that in our revocation
// example, if Bob uses his capability to Carol to obtain another capability from her, then send
// it back to her, the capability Carol receives back will NOT be revoked when Bob's access to
// Carol is revoked. Thus Bob can create long-term irrevocable connections. In most practical use
// cases, this is what you want. APIs commonly rely on the fact that a capability obtained and then
// passed back can be recognized as the original capability.
//
// Mark Miller on membranes: http://www.eros-os.org/pipermail/e-lang/2003-January/008434.html

#include "capability.h"

namespace capnp {

class MembranePolicy {
  // Applications may implement this interface to define a membrane policy, which allows some
  // calls crossing the membrane to be blocked or redirected.

public:
  virtual kj::Maybe<Capability::Client> inboundCall(
      uint64_t interfaceId, uint16_t methodId, Capability::Client target) = 0;
  // Given an inbound call (a call originating "outside" the membrane destined for an object
  // "inside" the membrane), decides what to do with it. The policy may:
  //
  // - Return null to indicate that the call should proceed to the destination. All capabilities
  //   in the parameters or result will be properly wrapped in the same membrane.
  // - Return a capability to have the call redirected to that capability. Note that the redirect
  //   capability will be treated as outside the membrane, so the params and results will not be
  //   auto-wrapped; however, the callee can easily wrap the returned capability in the membrane
  //   itself before returning to achieve this effect.
  // - Throw an exception to cause the call to fail with that exception.
  //
  // `target` is the underlying capability (*inside* the membrane) for which the call is destined.
  // Generally, the only way you should use `target` is to wrap it in some capability which you
  // return as a redirect. The redirect capability may modify the call in some way and send it to
  // `target`. Be careful to use `copyIntoMembrane()` and `copyOutOfMembrane()` as appropriate when
  // copying parameters or results across the membrane.
  //
  // Note that since `target` is inside the capability, if you were to directly return it (rather
  // than return null), the effect would be that the membrane would be broken: the call would
  // proceed directly and any new capabilities introduced through it would not be membraned. You
  // generally should not do that.

  virtual kj::Maybe<Capability::Client> outboundCall(
      uint64_t interfaceId, uint16_t methodId, Capability::Client target) = 0;
  // Like `inboundCall()`, but applies to calls originating *inside* the membrane and terminating
  // outside.
  //
  // Note: It is strongly recommended that `outboundCall()` returns null in exactly the same cases
  //   that `inboundCall()` return null. Conversely, for any case where `inboundCall()` would
  //   redirect or throw, `outboundCall()` should also redirect or throw. Otherwise, you can run
  //   into inconsistent behavion when a promise is returned across a membrane, and that promise
  //   later resolves to a capability on the other side of the membrane: calls on the promise
  //   will enter and then exit the membrane, but calls on the eventual resolution will not cross
  //   the membrane at all, so it is important that these two cases behave the same.

  virtual kj::Own<MembranePolicy> addRef() = 0;
  // Return a new owned pointer to the same policy.
  //
  // Typically an implementation of MembranePolicy should also inherit kj::Refcounted and implement
  // `addRef()` as `return kj::addRef(*this);`.
  //
  // Note that the membraning system considers two membranes created with the same MembranePolicy
  // object actually to be the *same* membrane. This is relevant when an object passes into the
  // membrane and then back out (or out and then back in): instead of double-wrapping the object,
  // the wrapping will be removed.
};

Capability::Client membrane(Capability::Client inner, kj::Own<MembranePolicy> policy);
// Wrap `inner` in a membrane specified by `policy`. `inner` is considered "inside" the membrane,
// while the returned capability should only be called from outside the membrane.

Capability::Client reverseMembrane(Capability::Client outer, kj::Own<MembranePolicy> policy);
// Like `membrane` but treat the input capability as "outside" the membrane, and return a
// capability appropriate for use inside.
//
// Applications typically won't use this directly; the membraning code automatically sets up
// reverse membranes where needed.

template <typename ClientType>
ClientType membrane(ClientType inner, kj::Own<MembranePolicy> policy);
template <typename ClientType>
ClientType reverseMembrane(ClientType inner, kj::Own<MembranePolicy> policy);
// Convenience templates which return the same interface type as the input.

template <typename ServerType>
typename ServerType::Serves::Client membrane(
    kj::Own<ServerType> inner, kj::Own<MembranePolicy> policy);
template <typename ServerType>
typename ServerType::Serves::Client reverseMembrane(
    kj::Own<ServerType> inner, kj::Own<MembranePolicy> policy);
// Convenience templates which input a capability server type and return the appropriate client
// type.

template <typename Reader>
Orphan<typename kj::Decay<Reader>::Reads> copyIntoMembrane(
    Reader&& from, Orphanage to, kj::Own<MembranePolicy> policy);
// Copy a Cap'n Proto object (e.g. struct or list), adding the given membrane to any capabilities
// found within it. `from` is interpreted as "outside" the membrane while `to` is "inside".

template <typename Reader>
Orphan<typename kj::Decay<Reader>::Reads> copyOutOfMembrane(
    Reader&& from, Orphanage to, kj::Own<MembranePolicy> policy);
// Like copyIntoMembrane() except that `from` is "inside" the membrane and `to` is "outside".

// =======================================================================================
// inline implementation details

template <typename ClientType>
ClientType membrane(ClientType inner, kj::Own<MembranePolicy> policy) {
  return membrane(Capability::Client(kj::mv(inner)), kj::mv(policy))
      .castAs<typename ClientType::Calls>();
}
template <typename ClientType>
ClientType reverseMembrane(ClientType inner, kj::Own<MembranePolicy> policy) {
  return reverseMembrane(Capability::Client(kj::mv(inner)), kj::mv(policy))
      .castAs<typename ClientType::Calls>();
}

template <typename ServerType>
typename ServerType::Serves::Client membrane(
    kj::Own<ServerType> inner, kj::Own<MembranePolicy> policy) {
  return membrane(Capability::Client(kj::mv(inner)), kj::mv(policy))
      .castAs<typename ServerType::Serves>();
}
template <typename ServerType>
typename ServerType::Serves::Client reverseMembrane(
    kj::Own<ServerType> inner, kj::Own<MembranePolicy> policy) {
  return reverseMembrane(Capability::Client(kj::mv(inner)), kj::mv(policy))
      .castAs<typename ServerType::Serves>();
}

namespace _ {  // private

OrphanBuilder copyOutOfMembrane(PointerReader from, Orphanage to,
                                kj::Own<MembranePolicy> policy, bool reverse);
OrphanBuilder copyOutOfMembrane(StructReader from, Orphanage to,
                                kj::Own<MembranePolicy> policy, bool reverse);
OrphanBuilder copyOutOfMembrane(ListReader from, Orphanage to,
                                kj::Own<MembranePolicy> policy, bool reverse);

}  // namespace _ (private)

template <typename Reader>
Orphan<typename kj::Decay<Reader>::Reads> copyIntoMembrane(
    Reader&& from, Orphanage to, kj::Own<MembranePolicy> policy) {
  return _::copyOutOfMembrane(
      _::PointerHelpers<typename kj::Decay<Reader>::Reads>::getInternalReader(from),
      to, kj::mv(policy), true);
}

template <typename Reader>
Orphan<typename kj::Decay<Reader>::Reads> copyOutOfMembrane(
    Reader&& from, Orphanage to, kj::Own<MembranePolicy> policy) {
  return _::copyOutOfMembrane(
      _::PointerHelpers<typename kj::Decay<Reader>::Reads>::getInternalReader(from),
      to, kj::mv(policy), false);
}

} // namespace capnp

#endif // CAPNP_MEMBRANE_H_
