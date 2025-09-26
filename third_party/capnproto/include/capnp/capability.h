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

#if CAPNP_LITE
#error "RPC APIs, including this header, are not available in lite mode."
#endif

#include <kj/async.h>
#include <kj/vector.h>
#include "raw-schema.h"
#include "any.h"
#include "pointer-helpers.h"

CAPNP_BEGIN_HEADER

namespace capnp {

template <typename Results>
class Response;

template <typename T>
class RemotePromise: public kj::Promise<Response<T>>, public T::Pipeline {
  // A Promise which supports pipelined calls.  T is typically a struct type.  T must declare
  // an inner "mix-in" type "Pipeline" which implements pipelining; RemotePromise simply
  // multiply-inherits that type along with Promise<Response<T>>.  T::Pipeline must be movable,
  // but does not need to be copyable (i.e. just like Promise<T>).
  //
  // The promise is for an owned pointer so that the RPC system can allocate the MessageReader
  // itself.

public:
  inline RemotePromise(kj::Promise<Response<T>>&& promise, typename T::Pipeline&& pipeline)
      : kj::Promise<Response<T>>(kj::mv(promise)),
        T::Pipeline(kj::mv(pipeline)) {}
  inline RemotePromise(decltype(nullptr))
      : kj::Promise<Response<T>>(nullptr),
        T::Pipeline(nullptr) {}
  KJ_DISALLOW_COPY(RemotePromise);
  RemotePromise(RemotePromise&& other) = default;
  RemotePromise& operator=(RemotePromise&& other) = default;

  kj::Promise<Response<T>> dropPipeline() {
    // Convenience method to convert this into a plain promise.
    return kj::mv(*this);
  }

  static RemotePromise<T> reducePromise(kj::Promise<RemotePromise>&& promise);
  // Hook for KJ so that Promise<RemotePromise<T>> automatically reduces to RemotePromise<T>.
};

class LocalClient;
namespace _ { // private
extern const RawSchema NULL_INTERFACE_SCHEMA;  // defined in schema.c++
class CapabilityServerSetBase;
struct PipelineBuilderPair;
}  // namespace _ (private)

struct Capability {
  // A capability without type-safe methods.  Typed capability clients wrap `Client` and typed
  // capability servers subclass `Server` to dispatch to the regular, typed methods.

  class Client;
  class Server;

  struct _capnpPrivate {
    struct IsInterface;
    static constexpr uint64_t typeId = 0x3;
    static constexpr Kind kind = Kind::INTERFACE;
    static constexpr _::RawSchema const* schema = &_::NULL_INTERFACE_SCHEMA;

    static const _::RawBrandedSchema* brand() {
      return &_::NULL_INTERFACE_SCHEMA.defaultBrand;
    }
  };
};

// =======================================================================================
// Capability clients

class RequestHook;
class ResponseHook;
class PipelineHook;
class ClientHook;
template <typename T>
class RevocableServer;

template <typename Params, typename Results>
class Request: public Params::Builder {
  // A call that hasn't been sent yet.  This class extends a Builder for the call's "Params"
  // structure with a method send() that actually sends it.
  //
  // Given a Cap'n Proto method `foo(a :A, b :B): C`, the generated client interface will have
  // a method `Request<FooParams, C> fooRequest()` (as well as a convenience method
  // `RemotePromise<C> foo(A::Reader a, B::Reader b)`).

public:
  inline Request(typename Params::Builder builder, kj::Own<RequestHook>&& hook)
      : Params::Builder(builder), hook(kj::mv(hook)) {}
  inline Request(decltype(nullptr)): Params::Builder(nullptr) {}

  RemotePromise<Results> send() KJ_WARN_UNUSED_RESULT;
  // Send the call and return a promise for the results.

  typename Results::Pipeline sendForPipeline();
  // Send the call in pipeline-only mode. The returned object can be used to make pipelined calls,
  // but there is no way to wait for the completion of the original call. This allows some
  // bookkeeping to be skipped under the hood, saving some time.
  //
  // Generally, this method should only be used when the caller will immediately make one or more
  // pipelined calls on the result, and then throw away the pipeline and all pipelined
  // capabilities. Other uses may run into caveats, such as:
  // - Normally, calling `whenResolved()` on a pipelined capability would wait for the original RPC
  //   to complete (and possibly other things, if that RPC itself returned a promise capability),
  //   but when using `sendPipelineOnly()`, `whenResolved()` may complete immediately, or never, or
  //   at an arbitrary time. Do not rely on it.
  // - Normal path shortening may not work with these capabilities. For exmaple, if the caller
  //   forwards a pipelined capability back to the callee's vat, calls made by the callee to that
  //   capability may continue to proxy through the caller. Conversely, if the callee ends up
  //   returning a capability that points back to the caller's vat, calls on the pipelined
  //   capability may continue to proxy through the callee.

private:
  kj::Own<RequestHook> hook;

  friend class Capability::Client;
  friend struct DynamicCapability;
  template <typename, typename>
  friend class CallContext;
  friend class RequestHook;
};

template <typename Params>
class StreamingRequest: public Params::Builder {
  // Like `Request` but for streaming requests.

public:
  inline StreamingRequest(typename Params::Builder builder, kj::Own<RequestHook>&& hook)
      : Params::Builder(builder), hook(kj::mv(hook)) {}
  inline StreamingRequest(decltype(nullptr)): Params::Builder(nullptr) {}

  kj::Promise<void> send() KJ_WARN_UNUSED_RESULT;

private:
  kj::Own<RequestHook> hook;

  friend class Capability::Client;
  friend struct DynamicCapability;
  template <typename, typename>
  friend class CallContext;
  friend class RequestHook;
};

template <typename Results>
class Response: public Results::Reader {
  // A completed call.  This class extends a Reader for the call's answer structure.  The Response
  // is move-only -- once it goes out-of-scope, the underlying message will be freed.

public:
  inline Response(typename Results::Reader reader, kj::Own<ResponseHook>&& hook)
      : Results::Reader(reader), hook(kj::mv(hook)) {}

private:
  kj::Own<ResponseHook> hook;

  template <typename, typename>
  friend class Request;
  friend class ResponseHook;
};

class Capability::Client {
  // Base type for capability clients.

public:
  typedef Capability Reads;
  typedef Capability Calls;

  Client(decltype(nullptr));
  // If you need to declare a Client before you have anything to assign to it (perhaps because
  // the assignment is going to occur in an if/else scope), you can start by initializing it to
  // `nullptr`.  The resulting client is not meant to be called and throws exceptions from all
  // methods.

  template <typename T, typename = kj::EnableIf<kj::canConvert<T*, Capability::Server*>()>>
  Client(kj::Own<T>&& server);
  // Make a client capability that wraps the given server capability.  The server's methods will
  // only be executed in the given EventLoop, regardless of what thread calls the client's methods.

  template <typename T, typename = kj::EnableIf<kj::canConvert<T*, Client*>()>>
  Client(kj::Promise<T>&& promise);
  // Make a client from a promise for a future client.  The resulting client queues calls until the
  // promise resolves.

  Client(kj::Exception&& exception);
  // Make a broken client that throws the given exception from all calls.

  Client(Client& other);
  Client& operator=(Client& other);
  // Copies by reference counting.  Warning:  This refcounting is not thread-safe.  All copies of
  // the client must remain in one thread.

  Client(Client&&) = default;
  Client& operator=(Client&&) = default;
  // Move constructor avoids reference counting.

  explicit Client(kj::Own<ClientHook>&& hook);
  // For use by the RPC implementation:  Wrap a ClientHook.

  template <typename T>
  typename T::Client castAs();
  // Reinterpret the capability as implementing the given interface.  Note that no error will occur
  // here if the capability does not actually implement this interface, but later method calls will
  // fail.  It's up to the application to decide how indicate that additional interfaces are
  // supported.
  //
  // TODO(perf):  GCC 4.8 / Clang 3.3:  rvalue-qualified version for better performance.

  template <typename T>
  typename T::Client castAs(InterfaceSchema schema);
  // Dynamic version.  `T` must be `DynamicCapability`, and you must `#include <capnp/dynamic.h>`.

  kj::Promise<void> whenResolved();
  // If the capability is actually only a promise, the returned promise resolves once the
  // capability itself has resolved to its final destination (or propagates the exception if
  // the capability promise is rejected).  This is mainly useful for error-checking in the case
  // where no calls are being made.  There is no reason to wait for this before making calls; if
  // the capability does not resolve, the call results will propagate the error.

  struct CallHints {
    bool noPromisePipelining = false;
    // Hints that the pipeline part of the VoidPromiseAndPipeline won't be used, so it can be
    // a bogus object.

    bool onlyPromisePipeline = false;
    // Hints that the promise part of the VoidPromiseAndPipeline won't be used, so it can be a
    // bogus promise.
    //
    // This hint is primarily intended to be passed to `ClientHook::call()`. When using
    // `ClientHook::newCall()`, you would instead indicate the hint by calling the `ResponseHook`'s
    // `sendForPipeline()` method. The effect of setting `onlyPromisePipeline = true` when invoking
    // `ClientHook::newCall()` is unspecified; it might cause the returned `Request` to support
    // only pipelining even when `send()` is called, or it might not.
  };

  Request<AnyPointer, AnyPointer> typelessRequest(
      uint64_t interfaceId, uint16_t methodId,
      kj::Maybe<MessageSize> sizeHint, CallHints hints);
  // Make a request without knowing the types of the params or results. You specify the type ID
  // and method number manually.

  kj::Promise<kj::Maybe<int>> getFd();
  // If the capability's server implemented Capability::Server::getFd() returning non-null, and all
  // RPC links between the client and server support FD passing, returns a file descriptor pointing
  // to the same underlying file description as the server did. Returns null if the server provided
  // no FD or if FD passing was unavailable at some intervening link.
  //
  // This returns a Promise to handle the case of an unresolved promise capability, e.g. a
  // pipelined capability. The promise resolves no later than when the capability settles, i.e.
  // the same time `whenResolved()` would complete.
  //
  // The file descriptor will remain open at least as long as the Capability::Client remains alive.
  // If you need it to last longer, you will need to `dup()` it.

  // TODO(someday):  method(s) for Join

protected:
  Client() = default;

  template <typename Params, typename Results>
  Request<Params, Results> newCall(uint64_t interfaceId, uint16_t methodId,
                                   kj::Maybe<MessageSize> sizeHint, CallHints hints);
  template <typename Params>
  StreamingRequest<Params> newStreamingCall(uint64_t interfaceId, uint16_t methodId,
                                            kj::Maybe<MessageSize> sizeHint, CallHints hints);

private:
  kj::Own<ClientHook> hook;

  static kj::Own<ClientHook> makeLocalClient(kj::Own<Capability::Server>&& server);
  static kj::Own<ClientHook> makeRevocableLocalClient(Capability::Server& server);
  static void revokeLocalClient(ClientHook& hook);
  static void revokeLocalClient(ClientHook& hook, kj::Exception&& reason);

  template <typename, Kind>
  friend struct _::PointerHelpers;
  friend struct DynamicCapability;
  friend class Orphanage;
  friend struct DynamicStruct;
  friend struct DynamicList;
  template <typename, Kind>
  friend struct List;
  friend class _::CapabilityServerSetBase;
  friend class ClientHook;
  template <typename T>
  friend class RevocableServer;
};

// =======================================================================================
// Capability servers

class CallContextHook;

template <typename Params, typename Results>
class CallContext: public kj::DisallowConstCopy {
  // Wrapper around CallContextHook with a specific return type.
  //
  // Methods of this class may only be called from within the server's event loop, not from other
  // threads.
  //
  // The CallContext becomes invalid as soon as the call reports completion.

public:
  explicit CallContext(CallContextHook& hook);

  typename Params::Reader getParams();
  // Get the params payload.

  void releaseParams();
  // Release the params payload.  getParams() will throw an exception after this is called.
  // Releasing the params may allow the RPC system to free up buffer space to handle other
  // requests.  Long-running asynchronous methods should try to call this as early as is
  // convenient.

  typename Results::Builder getResults(kj::Maybe<MessageSize> sizeHint = nullptr);
  typename Results::Builder initResults(kj::Maybe<MessageSize> sizeHint = nullptr);
  void setResults(typename Results::Reader value);
  void adoptResults(Orphan<Results>&& value);
  Orphanage getResultsOrphanage(kj::Maybe<MessageSize> sizeHint = nullptr);
  // Manipulate the results payload.  The "Return" message (part of the RPC protocol) will
  // typically be allocated the first time one of these is called.  Some RPC systems may
  // allocate these messages in a limited space (such as a shared memory segment), therefore the
  // application should delay calling these as long as is convenient to do so (but don't delay
  // if doing so would require extra copies later).
  //
  // `sizeHint` indicates a guess at the message size.  This will usually be used to decide how
  // much space to allocate for the first message segment (don't worry: only space that is actually
  // used will be sent on the wire).  If omitted, the system decides.  The message root pointer
  // should not be included in the size.  So, if you are simply going to copy some existing message
  // directly into the results, just call `.totalSize()` and pass that in.

  void setPipeline(typename Results::Pipeline&& pipeline);
  void setPipeline(typename Results::Pipeline& pipeline);
  // Tells the system where the capabilities in the response will eventually resolve to. This
  // allows requests that are promise-pipelined on this call's results to continue their journey
  // to the final destination before this call itself has completed.
  //
  // This is particularly useful when forwarding RPC calls to other remote servers, but where a
  // tail call can't be used. For example, imagine Alice calls `foo()` on Bob. In `foo()`'s
  // implementation, Bob calls `bar()` on Charlie. `bar()` returns a capability to Bob, and then
  // `foo()` returns the same capability on to Alice. Now imagine Alice is actually using promise
  // pipelining in a chain like `foo().getCap().baz()`. The `baz()` call will travel to Bob as a
  // pipelined call without waiting for `foo()` to return first. But once it gets to Bob, the
  // message has to patiently wait until `foo()` has completed there, before it can then be
  // forwarded on to Charlie. It would be better if immediately upon Bob calling `bar()` on
  // Charlie, then Alice's call to `baz()` could be forwarded to Charlie as a pipelined call,
  // without waiting for `bar()` to return. This would avoid a network round trip of latency
  // between Bob and Charlie.
  //
  // To solve this problem, Bob takes the pipeline object from the `bar()` call, transforms it into
  // an appropriate pipeline for a `foo()` call, and passes that to `setPipeline()`. This allows
  // Alice's pipelined `baz()` call to flow through immediately. The code looks like:
  //
  //     kj::Promise<void> foo(FooContext context) {
  //       auto barPromise = charlie.barRequest().send();
  //
  //       // Set up the final pipeline using pipelined capabilities from `barPromise`.
  //       capnp::PipelineBuilder<FooResults> pipeline;
  //       pipeline.setResultCap(barPromise.getSomeCap());
  //       context.setPipeline(pipeline.build());
  //
  //       // Now actually wait for the results and process them.
  //       return barPromise
  //           .then([context](capnp::Response<BarResults> response) mutable {
  //         auto results = context.initResults();
  //
  //         // Make sure to set up the capabilities exactly as we did in the pipeline.
  //         results.setResultCap(response.getSomeCap());
  //
  //         // ... do other stuff with the real response ...
  //       });
  //     }
  //
  // Of course, if `foo()` and `bar()` return exactly the same type, and Bob doesn't intend
  // to do anything with `bar()`'s response except pass it through, then `tailCall()` is a better
  // choice here. `setPipeline()` is useful when some transformation is needed on the response,
  // or the middleman needs to inspect the response for some reason.
  //
  // Note: This method has an overload that takes an lvalue reference for convenience. This
  //   overload increments the refcount on the underlying PipelineHook -- it does not keep the
  //   reference.
  //
  // Note: Capabilities returned by the replacement pipeline MUST either be exactly the same
  //   capabilities as in the final response, or eventually resolve to exactly the same
  //   capabilities, where "exactly the same" means the underlying `ClientHook` object is exactly
  //   the same object by identity. Resolving to some "equivalent" capability is not good enough.

  template <typename SubParams>
  kj::Promise<void> tailCall(Request<SubParams, Results>&& tailRequest);
  // Resolve the call by making a tail call.  `tailRequest` is a request that has been filled in
  // but not yet sent.  The context will send the call, then fill in the results with the result
  // of the call.  If tailCall() is used, {get,init,set,adopt}Results (above) *must not* be called.
  //
  // The RPC implementation may be able to optimize a tail call to another machine such that the
  // results never actually pass through this machine.  Even if no such optimization is possible,
  // `tailCall()` may allow pipelined calls to be forwarded optimistically to the new call site.
  //
  // In general, this should be the last thing a method implementation calls, and the promise
  // returned from `tailCall()` should then be returned by the method implementation.

  void allowCancellation()
      KJ_UNAVAILABLE(
          "As of Cap'n Proto 1.0, allowCancellation must be applied statically using an "
          "annotation in the schema. See annotations defined in /capnp/c++.capnp. For "
          "DynamicCapability::Server, use the constructor option (the annotation does not apply "
          "to DynamicCapability). This change was made to gain a significant performance boost -- "
          "dynamically allowing cancellation required excessive bookkeeping.");

private:
  CallContextHook* hook;

  friend class Capability::Server;
  friend struct DynamicCapability;
  friend class CallContextHook;
};

template <typename Params>
class StreamingCallContext: public kj::DisallowConstCopy {
  // Like CallContext but for streaming calls.

public:
  explicit StreamingCallContext(CallContextHook& hook);

  typename Params::Reader getParams();
  void releaseParams();

  // Note: tailCall() is not supported because:
  // - It would significantly complicate the implementation of streaming.
  // - It wouldn't be particularly useful since streaming calls don't return anything, and they
  //   already compensate for latency.

  void allowCancellation()
      KJ_UNAVAILABLE(
          "As of Cap'n Proto 1.0, allowCancellation must be applied statically using an "
          "annotation in the schema. See annotations defined in /capnp/c++.capnp. For "
          "DynamicCapability::Server, use the constructor option (the annotation does not apply "
          "to DynamicCapability). This change was made to gain a significant performance boost -- "
          "dynamically allowing cancellation required excessive bookkeeping.");

private:
  CallContextHook* hook;

  friend class Capability::Server;
  friend struct DynamicCapability;
  friend class CallContextHook;
};

class Capability::Server {
  // Objects implementing a Cap'n Proto interface must subclass this.  Typically, such objects
  // will instead subclass a typed Server interface which will take care of implementing
  // dispatchCall().

public:
  typedef Capability Serves;

  struct DispatchCallResult {
    kj::Promise<void> promise;
    // Promise for completion of the call.

    bool isStreaming;
    // If true, this method was declared as `-> stream;`. No other calls should be permitted until
    // this call finishes, and if this call throws an exception, all future calls will throw the
    // same exception.

    bool allowCancellation = false;
    // If true, the call can be canceled normally. If false, the immediate caller is responsible
    // for ensuring that cancellation is prevented and that `context` remains valid until the
    // call completes normally.
    //
    // See the `allowCancellation` annotation defined in `c++.capnp`.
  };

  virtual DispatchCallResult dispatchCall(uint64_t interfaceId, uint16_t methodId,
                                          CallContext<AnyPointer, AnyPointer> context) = 0;
  // Call the given method.  `params` is the input struct, and should be released as soon as it
  // is no longer needed.  `context` may be used to allocate the output struct and other call
  // logistics.

  virtual kj::Maybe<int> getFd() { return nullptr; }
  // If this capability is backed by a file descriptor that is safe to directly expose to clients,
  // returns that FD. When FD passing has been enabled in the RPC layer, this FD may be sent to
  // other processes along with the capability.

  virtual kj::Maybe<kj::Promise<Capability::Client>> shortenPath();
  // If this returns non-null, then it is a promise which, when resolved, points to a new
  // capability to which future calls can be sent. Use this in cases where an object implementation
  // might discover a more-optimized path some time after it starts.
  //
  // Implementing this (and returning non-null) will cause the capability to be advertised as a
  // promise at the RPC protocol level. Once the promise returned by shortenPath() resolves, the
  // remote client will receive a `Resolve` message updating it to point at the new destination.
  //
  // `shortenPath()` can also be used as a hack to shut up the client. If shortenPath() returns
  // a promise that resolves to an exception, then the client will be notified that the capability
  // is now broken. Assuming the client is using a correct RPC implemnetation, this should cause
  // all further calls initiated by the client to this capability to immediately fail client-side,
  // sparing the server's bandwidth.
  //
  // The default implementation always returns nullptr.

  // TODO(someday):  Method which can optionally be overridden to implement Join when the object is
  //   a proxy.

protected:
  inline Capability::Client thisCap();
  // Get a capability pointing to this object, much like the `this` keyword.
  //
  // The effect of this method is undefined if:
  // - No capability client has been created pointing to this object. (This is always the case in
  //   the server's constructor.)
  // - The capability client pointing at this object has been destroyed. (This is always the case
  //   in the server's destructor.)
  // - The capability client pointing at this object has been revoked using RevocableServer.
  // - Multiple capability clients have been created around the same server (possible if the server
  //   is refcounted, which is not recommended since the client itself provides refcounting).

  template <typename Params, typename Results>
  CallContext<Params, Results> internalGetTypedContext(
      CallContext<AnyPointer, AnyPointer> typeless);
  template <typename Params>
  StreamingCallContext<Params> internalGetTypedStreamingContext(
      CallContext<AnyPointer, AnyPointer> typeless);
  DispatchCallResult internalUnimplemented(const char* actualInterfaceName,
                                           uint64_t requestedTypeId);
  DispatchCallResult internalUnimplemented(const char* interfaceName,
                                           uint64_t typeId, uint16_t methodId);
  kj::Promise<void> internalUnimplemented(const char* interfaceName, const char* methodName,
                                          uint64_t typeId, uint16_t methodId);

private:
  ClientHook* thisHook = nullptr;
  friend class LocalClient;
};

template <typename T>
class RevocableServer {
  // Allows you to create a capability client pointing to a capability server without taking
  // ownership of the server. When `RevocableServer` is destroyed, all clients created through it
  // will become broken. All outstanding RPCs via those clients will be canceled and all future
  // RPCs will immediately throw. Hence, once the `RevocableServer` is destroyed, it is safe
  // to destroy the server object it referenced.
  //
  // This is particularly useful when you want to create a capability server that points to an
  // object that you do not own, and thus cannot keep alive beyond some defined lifetime. Since
  // you cannot force the client to respect lifetime rules, you should use a RevocableServer to
  // revoke access before the lifetime ends.
  //
  // The RevocableServer object can be moved (as long as the server outlives it).

public:
  RevocableServer(typename T::Server& server);
  RevocableServer(RevocableServer&&) = default;
  RevocableServer& operator=(RevocableServer&&) = default;
  ~RevocableServer() noexcept(false);
  KJ_DISALLOW_COPY(RevocableServer);

  typename T::Client getClient();

  void revoke();
  void revoke(kj::Exception&& reason);
  // Revokes the capability immediately, rather than waiting for the destructor. This can also
  // be used to specify a custom exception to use when revoking.

private:
  kj::Own<ClientHook> hook;
};

// =======================================================================================

template <typename T>
class PipelineBuilder: public T::Builder {
  // Convenience class to build a Pipeline object for use with CallContext::setPipeline().
  //
  // Building a pipeline object is like building an RPC result message, except that you only need
  // to fill in the capabilities, since the purpose is only to allow pipelined RPC requests to
  // flow through.
  //
  // See the docs for `CallContext::setPipeline()` for an example.

public:
  PipelineBuilder(uint firstSegmentWords = 64);
  // Construct a builder, allocating the given number of words for the first segment of the backing
  // message. Since `PipelineBuilder` is typically used with small RPC messages, the default size
  // here is considerably smaller than with MallocMessageBuilder.

  typename T::Pipeline build();
  // Constructs a `Pipeline` object backed by the current content of this builder. Calling this
  // consumes the `PipelineBuilder`; no further methods can be invoked.

private:
  kj::Own<PipelineHook> hook;

  PipelineBuilder(_::PipelineBuilderPair pair);
};

// =======================================================================================

class ReaderCapabilityTable: private _::CapTableReader {
  // Class which imbues Readers with the ability to read capabilities.
  //
  // In Cap'n Proto format, the encoding of a capability pointer is simply an integer index into
  // an external table. Since these pointers fundamentally point outside the message, a
  // MessageReader by default has no idea what they point at, and therefore reading capabilities
  // from such a reader will throw exceptions.
  //
  // In order to be able to read capabilities, you must first attach a capability table, using
  // this class. By "imbuing" a Reader, you get a new Reader which will interpret capability
  // pointers by treating them as indexes into the ReaderCapabilityTable.
  //
  // Note that when using Cap'n Proto's RPC system, this is handled automatically.

public:
  explicit ReaderCapabilityTable(kj::Array<kj::Maybe<kj::Own<ClientHook>>> table);
  KJ_DISALLOW_COPY_AND_MOVE(ReaderCapabilityTable);

  template <typename T>
  T imbue(T reader);
  // Return a reader equivalent to `reader` except that when reading capability-valued fields,
  // the capabilities are looked up in this table.

private:
  kj::Array<kj::Maybe<kj::Own<ClientHook>>> table;

  kj::Maybe<kj::Own<ClientHook>> extractCap(uint index) override;
};

class BuilderCapabilityTable: private _::CapTableBuilder {
  // Class which imbues Builders with the ability to read and write capabilities.
  //
  // This is much like ReaderCapabilityTable, except for builders. The table starts out empty,
  // but capabilities can be added to it over time.

public:
  BuilderCapabilityTable();
  KJ_DISALLOW_COPY_AND_MOVE(BuilderCapabilityTable);

  inline kj::ArrayPtr<kj::Maybe<kj::Own<ClientHook>>> getTable() { return table; }

  template <typename T>
  T imbue(T builder);
  // Return a builder equivalent to `builder` except that when reading capability-valued fields,
  // the capabilities are looked up in this table.

private:
  kj::Vector<kj::Maybe<kj::Own<ClientHook>>> table;

  kj::Maybe<kj::Own<ClientHook>> extractCap(uint index) override;
  uint injectCap(kj::Own<ClientHook>&& cap) override;
  void dropCap(uint index) override;
};

// =======================================================================================

namespace _ {  // private

class CapabilityServerSetBase {
public:
  Capability::Client addInternal(kj::Own<Capability::Server>&& server, void* ptr);
  kj::Promise<void*> getLocalServerInternal(Capability::Client& client);
};

}  // namespace _ (private)

template <typename T>
class CapabilityServerSet: private _::CapabilityServerSetBase {
  // Allows a server to recognize its own capabilities when passed back to it, and obtain the
  // underlying Server objects associated with them.
  //
  // All objects in the set must have the same interface type T. The objects may implement various
  // interfaces derived from T (and in fact T can be `capnp::Capability` to accept all objects),
  // but note that if you compile with RTTI disabled then you will not be able to down-cast through
  // virtual inheritance, and all inheritance between server interfaces is virtual. So, with RTTI
  // disabled, you will likely need to set T to be the most-derived Cap'n Proto interface type,
  // and you server class will need to be directly derived from that, so that you can use
  // static_cast (or kj::downcast) to cast to it after calling getLocalServer(). (If you compile
  // with RTTI, then you can freely dynamic_cast and ignore this issue!)

public:
  CapabilityServerSet() = default;
  KJ_DISALLOW_COPY_AND_MOVE(CapabilityServerSet);

  typename T::Client add(kj::Own<typename T::Server>&& server);
  // Create a new capability Client for the given Server and also add this server to the set.

  kj::Promise<kj::Maybe<typename T::Server&>> getLocalServer(typename T::Client& client);
  // Given a Client pointing to a server previously passed to add(), return the corresponding
  // Server. This returns a promise because if the input client is itself a promise, this must
  // wait for it to resolve. Keep in mind that the server will be deleted when all clients are
  // gone, so the caller should make sure to keep the client alive (hence why this method only
  // accepts an lvalue input).
};

// =======================================================================================
// Hook interfaces which must be implemented by the RPC system.  Applications never call these
// directly; the RPC system implements them and the types defined earlier in this file wrap them.

class RequestHook {
  // Hook interface implemented by RPC system representing a request being built.

public:
  virtual RemotePromise<AnyPointer> send() = 0;
  // Send the call and return a promise for the result.

  virtual kj::Promise<void> sendStreaming() = 0;
  // Send a streaming call.

  virtual AnyPointer::Pipeline sendForPipeline() = 0;
  // Send a call for pipelining purposes only.

  virtual const void* getBrand() = 0;
  // Returns a void* that identifies who made this request.  This can be used by an RPC adapter to
  // discover when tail call is going to be sent over its own connection and therefore can be
  // optimized into a remote tail call.

  template <typename T, typename U>
  inline static kj::Own<RequestHook> from(Request<T, U>&& request) {
    return kj::mv(request.hook);
  }
};

class ResponseHook {
  // Hook interface implemented by RPC system representing a response.
  //
  // At present this class has no methods.  It exists only for garbage collection -- when the
  // ResponseHook is destroyed, the results can be freed.

public:
  virtual ~ResponseHook() noexcept(false);
  // Just here to make sure the type is dynamic.

  template <typename T>
  inline static kj::Own<ResponseHook> from(Response<T>&& response) {
    return kj::mv(response.hook);
  }
};

// class PipelineHook is declared in any.h because it is needed there.

class ClientHook {
public:
  ClientHook();

  using CallHints = Capability::Client::CallHints;

  virtual Request<AnyPointer, AnyPointer> newCall(
      uint64_t interfaceId, uint16_t methodId, kj::Maybe<MessageSize> sizeHint,
      CallHints hints) = 0;
  // Start a new call, allowing the client to allocate request/response objects as it sees fit.
  // This version is used when calls are made from application code in the local process.

  struct VoidPromiseAndPipeline {
    kj::Promise<void> promise;
    kj::Own<PipelineHook> pipeline;
  };

  virtual VoidPromiseAndPipeline call(uint64_t interfaceId, uint16_t methodId,
                                      kj::Own<CallContextHook>&& context, CallHints hints) = 0;
  // Call the object, but the caller controls allocation of the request/response objects.  If the
  // callee insists on allocating these objects itself, it must make a copy.  This version is used
  // when calls come in over the network via an RPC system.  Note that even if the returned
  // `Promise<void>` is discarded, the call may continue executing if any pipelined calls are
  // waiting for it.
  //
  // The call must not begin synchronously; the callee must arrange for the call to begin in a
  // later turn of the event loop. Otherwise, application code may call back and affect the
  // callee's state in an unexpected way.

  virtual kj::Maybe<ClientHook&> getResolved() = 0;
  // If this ClientHook is a promise that has already resolved, returns the inner, resolved version
  // of the capability.  The caller may permanently replace this client with the resolved one if
  // desired.  Returns null if the client isn't a promise or hasn't resolved yet -- use
  // `whenMoreResolved()` to distinguish between them.
  //
  // Once a particular ClientHook's `getResolved()` returns non-null, it must permanently return
  // exactly the same resolution. This is why `getResolved()` returns a reference -- it is assumed
  // this object must have a strong reference to the resolution which it intends to keep
  // permanently, therefore the returned reference will live at least as long as this `ClientHook`.
  // This "only one resolution" policy is necessary for the RPC system to implement embargoes
  // properly.

  virtual kj::Maybe<kj::Promise<kj::Own<ClientHook>>> whenMoreResolved() = 0;
  // If this client is a settled reference (not a promise), return nullptr.  Otherwise, return a
  // promise that eventually resolves to a new client that is closer to being the final, settled
  // client (i.e. the value eventually returned by `getResolved()`).  Calling this repeatedly
  // should eventually produce a settled client.
  //
  // Once the promise resolves, `getResolved()` must return exactly the same `ClientHook` as the
  // one this Promise resolved to.

  kj::Promise<void> whenResolved();
  // Repeatedly calls whenMoreResolved() until it returns nullptr.

  virtual kj::Own<ClientHook> addRef() = 0;
  // Return a new reference to the same capability.

  virtual const void* getBrand() = 0;
  // Returns a void* that identifies who made this client.  This can be used by an RPC adapter to
  // discover when a capability it needs to marshal is one that it created in the first place, and
  // therefore it can transfer the capability without proxying.

  static const uint NULL_CAPABILITY_BRAND;
  static const uint BROKEN_CAPABILITY_BRAND;
  // Values are irrelevant; used for pointers.

  inline bool isNull() { return getBrand() == &NULL_CAPABILITY_BRAND; }
  // Returns true if the capability was created as a result of assigning a Client to null or by
  // reading a null pointer out of a Cap'n Proto message.

  inline bool isError() { return getBrand() == &BROKEN_CAPABILITY_BRAND; }
  // Returns true if the capability was created by newBrokenCap().

  virtual kj::Maybe<int> getFd() = 0;
  // Implements Capability::Client::getFd(). If this returns null but whenMoreResolved() returns
  // non-null, then Capability::Client::getFd() waits for resolution and tries again.

  static kj::Own<ClientHook> from(Capability::Client client) { return kj::mv(client.hook); }
};

class RevocableClientHook: public ClientHook {
public:
  virtual void revoke() = 0;
  virtual void revoke(kj::Exception&& reason) = 0;
};

class CallContextHook {
  // Hook interface implemented by RPC system to manage a call on the server side.  See
  // CallContext<T>.

public:
  virtual AnyPointer::Reader getParams() = 0;
  virtual void releaseParams() = 0;
  virtual AnyPointer::Builder getResults(kj::Maybe<MessageSize> sizeHint) = 0;
  virtual kj::Promise<void> tailCall(kj::Own<RequestHook>&& request) = 0;

  virtual void setPipeline(kj::Own<PipelineHook>&& pipeline) = 0;

  virtual kj::Promise<AnyPointer::Pipeline> onTailCall() = 0;
  // If `tailCall()` is called, resolves to the PipelineHook from the tail call.  An
  // implementation of `ClientHook::call()` is allowed to call this at most once.

  virtual ClientHook::VoidPromiseAndPipeline directTailCall(kj::Own<RequestHook>&& request) = 0;
  // Call this when you would otherwise call onTailCall() immediately followed by tailCall().
  // Implementations of tailCall() should typically call directTailCall() and then fulfill the
  // promise fulfiller for onTailCall() with the returned pipeline.

  virtual kj::Own<CallContextHook> addRef() = 0;

  template <typename Params, typename Results>
  static CallContextHook& from(CallContext<Params, Results>& context) { return *context.hook; }
  template <typename Params>
  static CallContextHook& from(StreamingCallContext<Params>& context) { return *context.hook; }
};

kj::Own<ClientHook> newLocalPromiseClient(kj::Promise<kj::Own<ClientHook>>&& promise);
// Returns a ClientHook that queues up calls until `promise` resolves, then forwards them to
// the new client.  This hook's `getResolved()` and `whenMoreResolved()` methods will reflect the
// redirection to the eventual replacement client.

kj::Own<PipelineHook> newLocalPromisePipeline(kj::Promise<kj::Own<PipelineHook>>&& promise);
// Returns a PipelineHook that queues up calls until `promise` resolves, then forwards them to
// the new pipeline.

kj::Own<ClientHook> newBrokenCap(kj::StringPtr reason);
kj::Own<ClientHook> newBrokenCap(kj::Exception&& reason);
// Helper function that creates a capability which simply throws exceptions when called.

kj::Own<PipelineHook> newBrokenPipeline(kj::Exception&& reason);
// Helper function that creates a pipeline which simply throws exceptions when called.

Request<AnyPointer, AnyPointer> newBrokenRequest(
    kj::Exception&& reason, kj::Maybe<MessageSize> sizeHint);
// Helper function that creates a Request object that simply throws exceptions when sent.

kj::Own<PipelineHook> getDisabledPipeline();
// Gets a PipelineHook appropriate to use when CallHints::noPromisePipelining is true. This will
// throw from all calls. This does not actually allocate the object; a static global object is
// returned with a null disposer.

// =======================================================================================
// Extend PointerHelpers for interfaces

namespace _ {  // private

template <typename T>
struct PointerHelpers<T, Kind::INTERFACE> {
  static inline typename T::Client get(PointerReader reader) {
    return typename T::Client(reader.getCapability());
  }
  static inline typename T::Client get(PointerBuilder builder) {
    return typename T::Client(builder.getCapability());
  }
  static inline void set(PointerBuilder builder, typename T::Client&& value) {
    builder.setCapability(kj::mv(value.Capability::Client::hook));
  }
  static inline void set(PointerBuilder builder, typename T::Client& value) {
    builder.setCapability(value.Capability::Client::hook->addRef());
  }
  static inline void adopt(PointerBuilder builder, Orphan<T>&& value) {
    builder.adopt(kj::mv(value.builder));
  }
  static inline Orphan<T> disown(PointerBuilder builder) {
    return Orphan<T>(builder.disown());
  }
};

}  // namespace _ (private)

// =======================================================================================
// Extend List for interfaces

template <typename T>
struct List<T, Kind::INTERFACE> {
  List() = delete;

  class Reader {
  public:
    typedef List<T> Reads;

    Reader() = default;
    inline explicit Reader(_::ListReader reader): reader(reader) {}

    inline uint size() const { return unbound(reader.size() / ELEMENTS); }
    inline typename T::Client operator[](uint index) const {
      KJ_IREQUIRE(index < size());
      return typename T::Client(reader.getPointerElement(
          bounded(index) * ELEMENTS).getCapability());
    }

    typedef _::IndexingIterator<const Reader, typename T::Client> Iterator;
    inline Iterator begin() const { return Iterator(this, 0); }
    inline Iterator end() const { return Iterator(this, size()); }

    inline MessageSize totalSize() const {
      return reader.totalSize().asPublic();
    }

  private:
    _::ListReader reader;
    template <typename U, Kind K>
    friend struct _::PointerHelpers;
    template <typename U, Kind K>
    friend struct List;
    friend class Orphanage;
    template <typename U, Kind K>
    friend struct ToDynamic_;
  };

  class Builder {
  public:
    typedef List<T> Builds;

    Builder() = delete;
    inline Builder(decltype(nullptr)) {}
    inline explicit Builder(_::ListBuilder builder): builder(builder) {}

    inline operator Reader() const { return Reader(builder.asReader()); }
    inline Reader asReader() const { return Reader(builder.asReader()); }

    inline uint size() const { return unbound(builder.size() / ELEMENTS); }
    inline typename T::Client operator[](uint index) {
      KJ_IREQUIRE(index < size());
      return typename T::Client(builder.getPointerElement(
          bounded(index) * ELEMENTS).getCapability());
    }
    inline void set(uint index, typename T::Client value) {
      KJ_IREQUIRE(index < size());
      builder.getPointerElement(bounded(index) * ELEMENTS).setCapability(kj::mv(value.hook));
    }
    inline void adopt(uint index, Orphan<T>&& value) {
      KJ_IREQUIRE(index < size());
      builder.getPointerElement(bounded(index) * ELEMENTS).adopt(kj::mv(value));
    }
    inline Orphan<T> disown(uint index) {
      KJ_IREQUIRE(index < size());
      return Orphan<T>(builder.getPointerElement(bounded(index) * ELEMENTS).disown());
    }

    typedef _::IndexingIterator<Builder, typename T::Client> Iterator;
    inline Iterator begin() { return Iterator(this, 0); }
    inline Iterator end() { return Iterator(this, size()); }

  private:
    _::ListBuilder builder;
    friend class Orphanage;
    template <typename U, Kind K>
    friend struct ToDynamic_;
  };

private:
  inline static _::ListBuilder initPointer(_::PointerBuilder builder, uint size) {
    return builder.initList(ElementSize::POINTER, bounded(size) * ELEMENTS);
  }
  inline static _::ListBuilder getFromPointer(_::PointerBuilder builder, const word* defaultValue) {
    return builder.getList(ElementSize::POINTER, defaultValue);
  }
  inline static _::ListReader getFromPointer(
      const _::PointerReader& reader, const word* defaultValue) {
    return reader.getList(ElementSize::POINTER, defaultValue);
  }

  template <typename U, Kind k>
  friend struct List;
  template <typename U, Kind K>
  friend struct _::PointerHelpers;
};

// =======================================================================================
// Inline implementation details

template <typename T>
RemotePromise<T> RemotePromise<T>::reducePromise(kj::Promise<RemotePromise>&& promise) {
  kj::Tuple<kj::Promise<Response<T>>, kj::Promise<kj::Own<PipelineHook>>> splitPromise =
      promise.then([](RemotePromise&& inner) {
    // `inner` is multiply-inherited, and we want to move away each superclass separately.
    // Let's create two references to make clear what we're doing (though this is not strictly
    // necessary).
    kj::Promise<Response<T>>& innerPromise = inner;
    typename T::Pipeline& innerPipeline = inner;
    return kj::tuple(kj::mv(innerPromise), PipelineHook::from(kj::mv(innerPipeline)));
  }).split();

  return RemotePromise(kj::mv(kj::get<0>(splitPromise)),
      typename T::Pipeline(AnyPointer::Pipeline(
          newLocalPromisePipeline(kj::mv(kj::get<1>(splitPromise))))));
}

template <typename Params, typename Results>
RemotePromise<Results> Request<Params, Results>::send() {
  auto typelessPromise = hook->send();
  hook = nullptr;  // prevent reuse

  // Convert the Promise to return the correct response type.
  // Explicitly upcast to kj::Promise to make clear that calling .then() doesn't invalidate the
  // Pipeline part of the RemotePromise.
  auto typedPromise = kj::implicitCast<kj::Promise<Response<AnyPointer>>&>(typelessPromise)
      .then([](Response<AnyPointer>&& response) -> Response<Results> {
        return Response<Results>(response.getAs<Results>(), kj::mv(response.hook));
      });

  // Wrap the typeless pipeline in a typed wrapper.
  typename Results::Pipeline typedPipeline(
      kj::mv(kj::implicitCast<AnyPointer::Pipeline&>(typelessPromise)));

  return RemotePromise<Results>(kj::mv(typedPromise), kj::mv(typedPipeline));
}

template <typename Params, typename Results>
typename Results::Pipeline Request<Params, Results>::sendForPipeline() {
  auto typelessPipeline = hook->sendForPipeline();
  hook = nullptr;  // prevent reuse
  return typename Results::Pipeline(kj::mv(typelessPipeline));
}

template <typename Params>
kj::Promise<void> StreamingRequest<Params>::send() {
  auto promise = hook->sendStreaming();
  hook = nullptr;  // prevent reuse
  return promise;
}

inline Capability::Client::Client(kj::Own<ClientHook>&& hook): hook(kj::mv(hook)) {}
template <typename T, typename>
inline Capability::Client::Client(kj::Own<T>&& server)
    : hook(makeLocalClient(kj::mv(server))) {}
template <typename T, typename>
inline Capability::Client::Client(kj::Promise<T>&& promise)
    : hook(newLocalPromiseClient(promise.then([](T&& t) { return kj::mv(t.hook); }))) {}
inline Capability::Client::Client(Client& other): hook(other.hook->addRef()) {}
inline Capability::Client& Capability::Client::operator=(Client& other) {
  hook = other.hook->addRef();
  return *this;
}
template <typename T>
inline typename T::Client Capability::Client::castAs() {
  return typename T::Client(hook->addRef());
}
inline Request<AnyPointer, AnyPointer> Capability::Client::typelessRequest(
    uint64_t interfaceId, uint16_t methodId,
    kj::Maybe<MessageSize> sizeHint, CallHints hints) {
  return newCall<AnyPointer, AnyPointer>(interfaceId, methodId, sizeHint, hints);
}
template <typename Params, typename Results>
inline Request<Params, Results> Capability::Client::newCall(
    uint64_t interfaceId, uint16_t methodId, kj::Maybe<MessageSize> sizeHint, CallHints hints) {
  auto typeless = hook->newCall(interfaceId, methodId, sizeHint, hints);
  return Request<Params, Results>(typeless.template getAs<Params>(), kj::mv(typeless.hook));
}
template <typename Params>
inline StreamingRequest<Params> Capability::Client::newStreamingCall(
    uint64_t interfaceId, uint16_t methodId, kj::Maybe<MessageSize> sizeHint, CallHints hints) {
  auto typeless = hook->newCall(interfaceId, methodId, sizeHint, hints);
  return StreamingRequest<Params>(typeless.template getAs<Params>(), kj::mv(typeless.hook));
}

template <typename Params, typename Results>
inline CallContext<Params, Results>::CallContext(CallContextHook& hook): hook(&hook) {}
template <typename Params>
inline StreamingCallContext<Params>::StreamingCallContext(CallContextHook& hook): hook(&hook) {}
template <typename Params, typename Results>
inline typename Params::Reader CallContext<Params, Results>::getParams() {
  return hook->getParams().template getAs<Params>();
}
template <typename Params>
inline typename Params::Reader StreamingCallContext<Params>::getParams() {
  return hook->getParams().template getAs<Params>();
}
template <typename Params, typename Results>
inline void CallContext<Params, Results>::releaseParams() {
  hook->releaseParams();
}
template <typename Params>
inline void StreamingCallContext<Params>::releaseParams() {
  hook->releaseParams();
}
template <typename Params, typename Results>
inline typename Results::Builder CallContext<Params, Results>::getResults(
    kj::Maybe<MessageSize> sizeHint) {
  // `template` keyword needed due to: http://llvm.org/bugs/show_bug.cgi?id=17401
  return hook->getResults(sizeHint).template getAs<Results>();
}
template <typename Params, typename Results>
inline typename Results::Builder CallContext<Params, Results>::initResults(
    kj::Maybe<MessageSize> sizeHint) {
  // `template` keyword needed due to: http://llvm.org/bugs/show_bug.cgi?id=17401
  return hook->getResults(sizeHint).template initAs<Results>();
}
template <typename Params, typename Results>
inline void CallContext<Params, Results>::setResults(typename Results::Reader value) {
  hook->getResults(value.totalSize()).template setAs<Results>(value);
}
template <typename Params, typename Results>
inline void CallContext<Params, Results>::adoptResults(Orphan<Results>&& value) {
  hook->getResults(nullptr).adopt(kj::mv(value));
}
template <typename Params, typename Results>
inline Orphanage CallContext<Params, Results>::getResultsOrphanage(
    kj::Maybe<MessageSize> sizeHint) {
  return Orphanage::getForMessageContaining(hook->getResults(sizeHint));
}
template <typename Params, typename Results>
void CallContext<Params, Results>::setPipeline(typename Results::Pipeline&& pipeline) {
  hook->setPipeline(PipelineHook::from(kj::mv(pipeline)));
}
template <typename Params, typename Results>
void CallContext<Params, Results>::setPipeline(typename Results::Pipeline& pipeline) {
  hook->setPipeline(PipelineHook::from(pipeline).addRef());
}
template <typename Params, typename Results>
template <typename SubParams>
inline kj::Promise<void> CallContext<Params, Results>::tailCall(
    Request<SubParams, Results>&& tailRequest) {
  return hook->tailCall(kj::mv(tailRequest.hook));
}

template <typename Params, typename Results>
CallContext<Params, Results> Capability::Server::internalGetTypedContext(
    CallContext<AnyPointer, AnyPointer> typeless) {
  return CallContext<Params, Results>(*typeless.hook);
}

template <typename Params>
StreamingCallContext<Params> Capability::Server::internalGetTypedStreamingContext(
    CallContext<AnyPointer, AnyPointer> typeless) {
  return StreamingCallContext<Params>(*typeless.hook);
}

Capability::Client Capability::Server::thisCap() {
  return Client(thisHook->addRef());
}

template <typename T>
RevocableServer<T>::RevocableServer(typename T::Server& server)
    : hook(Capability::Client::makeRevocableLocalClient(server)) {}
template <typename T>
RevocableServer<T>::~RevocableServer() noexcept(false) {
  // Check if moved away.
  if (hook.get() != nullptr) {
    Capability::Client::revokeLocalClient(*hook);
  }
}

template <typename T>
typename T::Client RevocableServer<T>::getClient() {
  return typename T::Client(hook->addRef());
}

template <typename T>
void RevocableServer<T>::revoke() {
  Capability::Client::revokeLocalClient(*hook);
}
template <typename T>
void RevocableServer<T>::revoke(kj::Exception&& exception) {
  Capability::Client::revokeLocalClient(*hook, kj::mv(exception));
}

namespace _ { // private

struct PipelineBuilderPair {
  AnyPointer::Builder root;
  kj::Own<PipelineHook> hook;
};

PipelineBuilderPair newPipelineBuilder(uint firstSegmentWords);

}  // namespace _ (private)

template <typename T>
PipelineBuilder<T>::PipelineBuilder(uint firstSegmentWords)
    : PipelineBuilder(_::newPipelineBuilder(firstSegmentWords)) {}

template <typename T>
PipelineBuilder<T>::PipelineBuilder(_::PipelineBuilderPair pair)
    : T::Builder(pair.root.initAs<T>()),
      hook(kj::mv(pair.hook)) {}

template <typename T>
typename T::Pipeline PipelineBuilder<T>::build() {
  // Prevent subsequent accidental modification. A good compiler should be able to optimize this
  // assignment away assuming the PipelineBuilder is not accessed again after this point.
  static_cast<typename T::Builder&>(*this) = nullptr;

  return typename T::Pipeline(AnyPointer::Pipeline(kj::mv(hook)));
}

template <typename T>
T ReaderCapabilityTable::imbue(T reader) {
  return T(_::PointerHelpers<FromReader<T>>::getInternalReader(reader).imbue(this));
}

template <typename T>
T BuilderCapabilityTable::imbue(T builder) {
  return T(_::PointerHelpers<FromBuilder<T>>::getInternalBuilder(kj::mv(builder)).imbue(this));
}

template <typename T>
typename T::Client CapabilityServerSet<T>::add(kj::Own<typename T::Server>&& server) {
  void* ptr = reinterpret_cast<void*>(server.get());
  // Clang insists that `castAs` is a template-dependent member and therefore we need the
  // `template` keyword here, but AFAICT this is wrong: addImpl() is not a template.
  return addInternal(kj::mv(server), ptr).template castAs<T>();
}

template <typename T>
kj::Promise<kj::Maybe<typename T::Server&>> CapabilityServerSet<T>::getLocalServer(
    typename T::Client& client) {
  return getLocalServerInternal(client)
      .then([](void* server) -> kj::Maybe<typename T::Server&> {
    if (server == nullptr) {
      return nullptr;
    } else {
      return *reinterpret_cast<typename T::Server*>(server);
    }
  });
}

template <typename T>
struct Orphanage::GetInnerReader<T, Kind::INTERFACE> {
  static inline kj::Own<ClientHook> apply(typename T::Client t) {
    return ClientHook::from(kj::mv(t));
  }
};

#define CAPNP_CAPABILITY_H_INCLUDED  // for testing includes in unit test

}  // namespace capnp

CAPNP_END_HEADER
