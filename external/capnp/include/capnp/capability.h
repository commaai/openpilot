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

#ifndef CAPNP_CAPABILITY_H_
#define CAPNP_CAPABILITY_H_

#if defined(__GNUC__) && !defined(CAPNP_HEADER_WARNINGS)
#pragma GCC system_header
#endif

#if CAPNP_LITE
#error "RPC APIs, including this header, are not available in lite mode."
#endif

#include <kj/async.h>
#include <kj/vector.h>
#include "raw-schema.h"
#include "any.h"
#include "pointer-helpers.h"

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
};

class LocalClient;
namespace _ { // private
extern const RawSchema NULL_INTERFACE_SCHEMA;  // defined in schema.c++
class CapabilityServerSetBase;
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

  Request<AnyPointer, AnyPointer> typelessRequest(
      uint64_t interfaceId, uint16_t methodId,
      kj::Maybe<MessageSize> sizeHint);
  // Make a request without knowing the types of the params or results. You specify the type ID
  // and method number manually.

  // TODO(someday):  method(s) for Join

protected:
  Client() = default;

  template <typename Params, typename Results>
  Request<Params, Results> newCall(uint64_t interfaceId, uint16_t methodId,
                                   kj::Maybe<MessageSize> sizeHint);

private:
  kj::Own<ClientHook> hook;

  static kj::Own<ClientHook> makeLocalClient(kj::Own<Capability::Server>&& server);

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

  void allowCancellation();
  // Indicate that it is OK for the RPC system to discard its Promise for this call's result if
  // the caller cancels the call, thereby transitively canceling any asynchronous operations the
  // call implementation was performing.  This is not done by default because it could represent a
  // security risk:  applications must be carefully written to ensure that they do not end up in
  // a bad state if an operation is canceled at an arbitrary point.  However, for long-running
  // method calls that hold significant resources, prompt cancellation is often useful.
  //
  // Keep in mind that asynchronous cancellation cannot occur while the method is synchronously
  // executing on a local thread.  The method must perform an asynchronous operation or call
  // `EventLoop::current().evalLater()` to yield control.
  //
  // Note:  You might think that we should offer `onCancel()` and/or `isCanceled()` methods that
  // provide notification when the caller cancels the request without forcefully killing off the
  // promise chain.  Unfortunately, this composes poorly with promise forking:  the canceled
  // path may be just one branch of a fork of the result promise.  The other branches still want
  // the call to continue.  Promise forking is used within the Cap'n Proto implementation -- in
  // particular each pipelined call forks the result promise.  So, if a caller made a pipelined
  // call and then dropped the original object, the call should not be canceled, but it would be
  // excessively complicated for the framework to avoid notififying of cancellation as long as
  // pipelined calls still exist.

private:
  CallContextHook* hook;

  friend class Capability::Server;
  friend struct DynamicCapability;
};

class Capability::Server {
  // Objects implementing a Cap'n Proto interface must subclass this.  Typically, such objects
  // will instead subclass a typed Server interface which will take care of implementing
  // dispatchCall().

public:
  typedef Capability Serves;

  virtual kj::Promise<void> dispatchCall(uint64_t interfaceId, uint16_t methodId,
                                         CallContext<AnyPointer, AnyPointer> context) = 0;
  // Call the given method.  `params` is the input struct, and should be released as soon as it
  // is no longer needed.  `context` may be used to allocate the output struct and deal with
  // cancellation.

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
  // - Multiple capability clients have been created around the same server (possible if the server
  //   is refcounted, which is not recommended since the client itself provides refcounting).

  template <typename Params, typename Results>
  CallContext<Params, Results> internalGetTypedContext(
      CallContext<AnyPointer, AnyPointer> typeless);
  kj::Promise<void> internalUnimplemented(const char* actualInterfaceName,
                                          uint64_t requestedTypeId);
  kj::Promise<void> internalUnimplemented(const char* interfaceName,
                                          uint64_t typeId, uint16_t methodId);
  kj::Promise<void> internalUnimplemented(const char* interfaceName, const char* methodName,
                                          uint64_t typeId, uint16_t methodId);

private:
  ClientHook* thisHook = nullptr;
  friend class LocalClient;
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
  KJ_DISALLOW_COPY(ReaderCapabilityTable);

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
  KJ_DISALLOW_COPY(BuilderCapabilityTable);

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
  KJ_DISALLOW_COPY(CapabilityServerSet);

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

  virtual Request<AnyPointer, AnyPointer> newCall(
      uint64_t interfaceId, uint16_t methodId, kj::Maybe<MessageSize> sizeHint) = 0;
  // Start a new call, allowing the client to allocate request/response objects as it sees fit.
  // This version is used when calls are made from application code in the local process.

  struct VoidPromiseAndPipeline {
    kj::Promise<void> promise;
    kj::Own<PipelineHook> pipeline;
  };

  virtual VoidPromiseAndPipeline call(uint64_t interfaceId, uint16_t methodId,
                                      kj::Own<CallContextHook>&& context) = 0;
  // Call the object, but the caller controls allocation of the request/response objects.  If the
  // callee insists on allocating these objects itself, it must make a copy.  This version is used
  // when calls come in over the network via an RPC system.  Note that even if the returned
  // `Promise<void>` is discarded, the call may continue executing if any pipelined calls are
  // waiting for it.
  //
  // Since the caller of this method chooses the CallContext implementation, it is the caller's
  // responsibility to ensure that the returned promise is not canceled unless allowed via
  // the context's `allowCancellation()`.
  //
  // The call must not begin synchronously; the callee must arrange for the call to begin in a
  // later turn of the event loop. Otherwise, application code may call back and affect the
  // callee's state in an unexpected way.

  virtual kj::Maybe<ClientHook&> getResolved() = 0;
  // If this ClientHook is a promise that has already resolved, returns the inner, resolved version
  // of the capability.  The caller may permanently replace this client with the resolved one if
  // desired.  Returns null if the client isn't a promise or hasn't resolved yet -- use
  // `whenMoreResolved()` to distinguish between them.

  virtual kj::Maybe<kj::Promise<kj::Own<ClientHook>>> whenMoreResolved() = 0;
  // If this client is a settled reference (not a promise), return nullptr.  Otherwise, return a
  // promise that eventually resolves to a new client that is closer to being the final, settled
  // client (i.e. the value eventually returned by `getResolved()`).  Calling this repeatedly
  // should eventually produce a settled client.

  kj::Promise<void> whenResolved();
  // Repeatedly calls whenMoreResolved() until it returns nullptr.

  virtual kj::Own<ClientHook> addRef() = 0;
  // Return a new reference to the same capability.

  virtual const void* getBrand() = 0;
  // Returns a void* that identifies who made this client.  This can be used by an RPC adapter to
  // discover when a capability it needs to marshal is one that it created in the first place, and
  // therefore it can transfer the capability without proxying.

  static const uint NULL_CAPABILITY_BRAND;
  // Value is irrelevant; used for pointer.

  inline bool isNull() { return getBrand() == &NULL_CAPABILITY_BRAND; }
  // Returns true if the capability was created as a result of assigning a Client to null or by
  // reading a null pointer out of a Cap'n Proto message.

  virtual void* getLocalServer(_::CapabilityServerSetBase& capServerSet);
  // If this is a local capability created through `capServerSet`, return the underlying Server.
  // Otherwise, return nullptr. Default implementation (which everyone except LocalClient should
  // use) always returns nullptr.

  static kj::Own<ClientHook> from(Capability::Client client) { return kj::mv(client.hook); }
};

class CallContextHook {
  // Hook interface implemented by RPC system to manage a call on the server side.  See
  // CallContext<T>.

public:
  virtual AnyPointer::Reader getParams() = 0;
  virtual void releaseParams() = 0;
  virtual AnyPointer::Builder getResults(kj::Maybe<MessageSize> sizeHint) = 0;
  virtual kj::Promise<void> tailCall(kj::Own<RequestHook>&& request) = 0;
  virtual void allowCancellation() = 0;

  virtual kj::Promise<AnyPointer::Pipeline> onTailCall() = 0;
  // If `tailCall()` is called, resolves to the PipelineHook from the tail call.  An
  // implementation of `ClientHook::call()` is allowed to call this at most once.

  virtual ClientHook::VoidPromiseAndPipeline directTailCall(kj::Own<RequestHook>&& request) = 0;
  // Call this when you would otherwise call onTailCall() immediately followed by tailCall().
  // Implementations of tailCall() should typically call directTailCall() and then fulfill the
  // promise fulfiller for onTailCall() with the returned pipeline.

  virtual kj::Own<CallContextHook> addRef() = 0;
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
inline kj::Promise<void> Capability::Client::whenResolved() {
  return hook->whenResolved();
}
inline Request<AnyPointer, AnyPointer> Capability::Client::typelessRequest(
    uint64_t interfaceId, uint16_t methodId,
    kj::Maybe<MessageSize> sizeHint) {
  return newCall<AnyPointer, AnyPointer>(interfaceId, methodId, sizeHint);
}
template <typename Params, typename Results>
inline Request<Params, Results> Capability::Client::newCall(
    uint64_t interfaceId, uint16_t methodId, kj::Maybe<MessageSize> sizeHint) {
  auto typeless = hook->newCall(interfaceId, methodId, sizeHint);
  return Request<Params, Results>(typeless.template getAs<Params>(), kj::mv(typeless.hook));
}

template <typename Params, typename Results>
inline CallContext<Params, Results>::CallContext(CallContextHook& hook): hook(&hook) {}
template <typename Params, typename Results>
inline typename Params::Reader CallContext<Params, Results>::getParams() {
  return hook->getParams().template getAs<Params>();
}
template <typename Params, typename Results>
inline void CallContext<Params, Results>::releaseParams() {
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
template <typename SubParams>
inline kj::Promise<void> CallContext<Params, Results>::tailCall(
    Request<SubParams, Results>&& tailRequest) {
  return hook->tailCall(kj::mv(tailRequest.hook));
}
template <typename Params, typename Results>
inline void CallContext<Params, Results>::allowCancellation() {
  hook->allowCancellation();
}

template <typename Params, typename Results>
CallContext<Params, Results> Capability::Server::internalGetTypedContext(
    CallContext<AnyPointer, AnyPointer> typeless) {
  return CallContext<Params, Results>(*typeless.hook);
}

Capability::Client Capability::Server::thisCap() {
  return Client(thisHook->addRef());
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

}  // namespace capnp

#endif  // CAPNP_CAPABILITY_H_
