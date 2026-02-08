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

// This file contains extended inline implementation details that are required along with async.h.
// We move this all into a separate file to make async.h more readable.
//
// Non-inline declarations here are defined in async.c++.

#pragma once

#ifndef KJ_ASYNC_H_INCLUDED
#error "Do not include this directly; include kj/async.h."
#include "async.h"  // help IDE parse this file
#endif

#if _MSC_VER && KJ_HAS_COROUTINE
#include <intrin.h>
#endif

#include <kj/list.h>

KJ_BEGIN_HEADER

namespace kj {
namespace _ {  // private

template <typename T>
class ExceptionOr;

class ExceptionOrValue {
public:
  ExceptionOrValue(bool, Exception&& exception): exception(kj::mv(exception)) {}
  KJ_DISALLOW_COPY(ExceptionOrValue);

  void addException(Exception&& exception) {
    if (this->exception == nullptr) {
      this->exception = kj::mv(exception);
    }
  }

  template <typename T>
  ExceptionOr<T>& as() { return *static_cast<ExceptionOr<T>*>(this); }
  template <typename T>
  const ExceptionOr<T>& as() const { return *static_cast<const ExceptionOr<T>*>(this); }

  Maybe<Exception> exception;

protected:
  // Allow subclasses to have move constructor / assignment.
  ExceptionOrValue() = default;
  ExceptionOrValue(ExceptionOrValue&& other) = default;
  ExceptionOrValue& operator=(ExceptionOrValue&& other) = default;
};

template <typename T>
class ExceptionOr: public ExceptionOrValue {
public:
  ExceptionOr() = default;
  ExceptionOr(T&& value): value(kj::mv(value)) {}
  ExceptionOr(bool, Exception&& exception): ExceptionOrValue(false, kj::mv(exception)) {}
  ExceptionOr(ExceptionOr&&) = default;
  ExceptionOr& operator=(ExceptionOr&&) = default;

  Maybe<T> value;
};

template <typename T>
inline T convertToReturn(ExceptionOr<T>&& result) {
  KJ_IF_MAYBE(value, result.value) {
    KJ_IF_MAYBE(exception, result.exception) {
      throwRecoverableException(kj::mv(*exception));
    }
    return _::returnMaybeVoid(kj::mv(*value));
  } else KJ_IF_MAYBE(exception, result.exception) {
    throwFatalException(kj::mv(*exception));
  } else {
    // Result contained neither a value nor an exception?
    KJ_UNREACHABLE;
  }
}

inline void convertToReturn(ExceptionOr<Void>&& result) {
  // Override <void> case to use throwRecoverableException().

  if (result.value != nullptr) {
    KJ_IF_MAYBE(exception, result.exception) {
      throwRecoverableException(kj::mv(*exception));
    }
  } else KJ_IF_MAYBE(exception, result.exception) {
    throwRecoverableException(kj::mv(*exception));
  } else {
    // Result contained neither a value nor an exception?
    KJ_UNREACHABLE;
  }
}

class TraceBuilder {
  // Helper for methods that build a call trace.
public:
  TraceBuilder(ArrayPtr<void*> space)
      : start(space.begin()), current(space.begin()), limit(space.end()) {}

  inline void add(void* addr) {
    if (current < limit) {
      *current++ = addr;
    }
  }

  inline bool full() const { return current == limit; }

  ArrayPtr<void*> finish() {
    return arrayPtr(start, current);
  }

  String toString();

private:
  void** start;
  void** current;
  void** limit;
};

struct alignas(void*) PromiseArena {
  // Space in which a chain of promises may be allocated. See PromiseDisposer.
  byte bytes[1024];
};

class Event: private AsyncObject {
  // An event waiting to be executed.  Not for direct use by applications -- promises use this
  // internally.

public:
  Event(SourceLocation location);
  Event(kj::EventLoop& loop, SourceLocation location);
  ~Event() noexcept(false);
  KJ_DISALLOW_COPY_AND_MOVE(Event);

  void armDepthFirst();
  // Enqueue this event so that `fire()` will be called from the event loop soon.
  //
  // Events scheduled in this way are executed in depth-first order:  if an event callback arms
  // more events, those events are placed at the front of the queue (in the order in which they
  // were armed), so that they run immediately after the first event's callback returns.
  //
  // Depth-first event scheduling is appropriate for events that represent simple continuations
  // of a previous event that should be globbed together for performance.  Depth-first scheduling
  // can lead to starvation, so any long-running task must occasionally yield with
  // `armBreadthFirst()`.  (Promise::then() uses depth-first whereas evalLater() uses
  // breadth-first.)
  //
  // To use breadth-first scheduling instead, use `armBreadthFirst()`.

  void armBreadthFirst();
  // Like `armDepthFirst()` except that the event is placed at the end of the queue.

  void armLast();
  // Enqueues this event to happen after all other events have run to completion and there is
  // really nothing left to do except wait for I/O.

  bool isNext();
  // True if the Event has been armed and is next in line to be fired. This can be used after
  // calling PromiseNode::onReady(event) to determine if a promise being waited is immediately
  // ready, in which case continuations may be optimistically run without returning to the event
  // loop. Note that this optimization is only valid if we know that we would otherwise immediately
  // return to the event loop without running more application code. So this turns out to be useful
  // in fairly narrow circumstances, chiefly when a coroutine is about to suspend, but discovers it
  // doesn't need to.
  //
  // Returns false if the event loop is not currently running. This ensures that promise
  // continuations don't execute except under a call to .wait().

  void disarm();
  // If the event is armed but hasn't fired, cancel it. (Destroying the event does this
  // implicitly.)

  virtual void traceEvent(TraceBuilder& builder) = 0;
  // Build a trace of the callers leading up to this event. `builder` will be populated with
  // "return addresses" of the promise chain waiting on this event. The return addresses may
  // actually the addresses of lambdas passed to .then(), but in any case, feeding them into
  // addr2line should produce useful source code locations.
  //
  // `traceEvent()` may be called from an async signal handler while `fire()` is executing. It
  // must not allocate nor take locks.

  String traceEvent();
  // Helper that builds a trace and stringifies it.

protected:
  virtual Maybe<Own<Event>> fire() = 0;
  // Fire the event.  Possibly returns a pointer to itself, which will be discarded by the
  // caller.  This is the only way that an event can delete itself as a result of firing, as
  // doing so from within fire() will throw an exception.

private:
  friend class kj::EventLoop;
  EventLoop& loop;
  Event* next;
  Event** prev;
  bool firing = false;

  static constexpr uint MAGIC_LIVE_VALUE = 0x1e366381u;
  uint live = MAGIC_LIVE_VALUE;
  SourceLocation location;
};

class PromiseArenaMember {
  // An object that is allocated in a PromiseArena. `PromiseNode` inherits this, and most
  // arena-allocated objects are `PromiseNode` subclasses, but `TaskSet::Task`, ForkHub, and
  // potentially other objects that commonly live on the end of a promise chain can also leverage
  // this.

public:
  virtual void destroy() = 0;
  // Destroys and frees the node.
  //
  // If the node was allocated using allocPromise<T>(), then destroy() must call
  // freePromise<T>(this). If it was allocated some other way, then it is `destroy()`'s
  // responsibility to complete any necessary cleanup of memory, e.g. call `delete this`.
  //
  // We use this instead of a virtual destructor for two reasons:
  // 1. Coroutine nodes are not independent objects, they have to call destroy() on the coroutine
  //    handle to delete themselves.
  // 2. XThreadEvents sometimes leave it up to a different thread to actually delete the object.

private:
  PromiseArena* arena = nullptr;
  // If non-null, then this PromiseNode is the last node allocated within the given arena, and
  // therefore owns the arena. After this node is destroyed, the arena should be deleted.
  //
  // PromiseNodes are allocated within the arena starting from the end, and `PromiseNode`s
  // allocated this way are required to have `PromiseNode` itself as their leftmost inherited type,
  // so that the pointers match. Thus, the space in `arena` from its start to the location of the
  // `PromiseNode` is known to be available for subsequent allocations (which should then take
  // ownership of the arena).

  friend class PromiseDisposer;
};

class PromiseNode: public PromiseArenaMember, private AsyncObject {
  // A Promise<T> contains a chain of PromiseNodes tracking the pending transformations.
  //
  // To reduce generated code bloat, PromiseNode is not a template.  Instead, it makes very hacky
  // use of pointers to ExceptionOrValue which actually point to ExceptionOr<T>, but are only
  // so down-cast in the few places that really need to be templated.  Luckily this is all
  // internal implementation details.

public:
  virtual void onReady(Event* event) noexcept = 0;
  // Arms the given event when ready.
  //
  // May be called multiple times. If called again before the event was armed, the old event will
  // never be armed, only the new one. If called again after the event was armed, the new event
  // will be armed immediately. Can be called with nullptr to un-register the existing event.

  virtual void setSelfPointer(OwnPromiseNode* selfPtr) noexcept;
  // Tells the node that `selfPtr` is the pointer that owns this node, and will continue to own
  // this node until it is destroyed or setSelfPointer() is called again.  ChainPromiseNode uses
  // this to shorten redundant chains.  The default implementation does nothing; only
  // ChainPromiseNode should implement this.

  virtual void get(ExceptionOrValue& output) noexcept = 0;
  // Get the result.  `output` points to an ExceptionOr<T> into which the result will be written.
  // Can only be called once, and only after the node is ready.  Must be called directly from the
  // event loop, with no application code on the stack.

  virtual void tracePromise(TraceBuilder& builder, bool stopAtNextEvent) = 0;
  // Build a trace of this promise chain, showing what it is currently waiting on.
  //
  // Since traces are ordered callee-before-caller, PromiseNode::tracePromise() should typically
  // recurse to its child first, then after the child returns, add itself to the trace.
  //
  // If `stopAtNextEvent` is true, then the trace should stop as soon as it hits a PromiseNode that
  // also implements Event, and should not trace that node or its children. This is used in
  // conjunction with Event::traceEvent(). The chain of Events is often more sparse than the chain
  // of PromiseNodes, because a TransformPromiseNode (which implements .then()) is not itself an
  // Event. TransformPromiseNode instead tells its child node to directly notify its *parent* node
  // when it is ready, and then TransformPromiseNode applies the .then() transformation during the
  // call to .get().
  //
  // So, when we trace the chain of Events backwards, we end up hoping over segments of
  // TransformPromiseNodes (and other similar types). In order to get those added to the trace,
  // each Event must call back down the PromiseNode chain in the opposite direction, using this
  // method.
  //
  // `tracePromise()` may be called from an async signal handler while `get()` is executing. It
  // must not allocate nor take locks.

  template <typename T>
  static OwnPromiseNode from(T&& promise) {
    // Given a Promise, extract the PromiseNode.
    return kj::mv(promise.node);
  }
  template <typename T>
  static PromiseNode& from(T& promise) {
    // Given a Promise, extract the PromiseNode.
    return *promise.node;
  }
  template <typename T>
  static T to(OwnPromiseNode&& node) {
    // Construct a Promise from a PromiseNode. (T should be a Promise type.)
    return T(false, kj::mv(node));
  }

protected:
  class OnReadyEvent {
    // Helper class for implementing onReady().

  public:
    void init(Event* newEvent);

    void arm();
    void armBreadthFirst();
    // Arms the event if init() has already been called and makes future calls to init()
    // automatically arm the event.

    inline void traceEvent(TraceBuilder& builder) {
      if (event != nullptr && !builder.full()) event->traceEvent(builder);
    }

  private:
    Event* event = nullptr;
  };
};

class PromiseDisposer {
public:
  template <typename T>
  static constexpr bool canArenaAllocate() {
    // We can only use arena allocation for types that fit in an arena and have pointer-size
    // alignment. Anything else will need to be allocated as a separate heap object.
    return sizeof(T) <= sizeof(PromiseArena) && alignof(T) <= alignof(void*);
  }

  static void dispose(PromiseArenaMember* node) {
    PromiseArena* arena = node->arena;
    node->destroy();
    delete arena;  // reminder: `delete` automatically ignores null pointers
  }

  template <typename T, typename D = PromiseDisposer, typename... Params>
  static kj::Own<T, D> alloc(Params&&... params) noexcept {
    // Implements allocPromise().
    T* ptr;
    if (!canArenaAllocate<T>()) {
      // Node too big (or needs weird alignment), fall back to regular heap allocation.
      ptr = new T(kj::fwd<Params>(params)...);
    } else {
      // Start a new arena.
      //
      // NOTE: As in append() (below), we don't implement exception-safety because it causes code
      //   bloat and these constructors probably don't throw. Instead this function is noexcept, so
      //   if a constructor does throw, it'll crash rather than leak memory.
      auto* arena = new PromiseArena;
      ptr = reinterpret_cast<T*>(arena + 1) - 1;
      ctor(*ptr, kj::fwd<Params>(params)...);
      ptr->arena = arena;
      KJ_IREQUIRE(reinterpret_cast<void*>(ptr) ==
                  reinterpret_cast<void*>(static_cast<PromiseArenaMember*>(ptr)),
          "PromiseArenaMember must be the leftmost inherited type.");
    }
    return kj::Own<T, D>(ptr);
  }

  template <typename T, typename D = PromiseDisposer, typename... Params>
  static kj::Own<T, D> append(
      OwnPromiseNode&& next, Params&&... params) noexcept {
    // Implements appendPromise().

    PromiseArena* arena = next->arena;

    if (!canArenaAllocate<T>() || arena == nullptr ||
        reinterpret_cast<byte*>(next.get()) - reinterpret_cast<byte*>(arena) < sizeof(T)) {
      // No arena available, or not enough space, or weird alignment needed. Start new arena.
      return alloc<T, D>(kj::mv(next), kj::fwd<Params>(params)...);
    } else {
      // Append to arena.
      //
      // NOTE: When we call ctor(), it takes ownership of `next`, so we shouldn't assume `next`
      //   still exists after it returns. So we have to remove ownership of the arena before that.
      //   In theory if we wanted this to be exception-safe, we'd also have to arrange to delete
      //   the arena if the constructor throws. However, in practice none of the PromiseNode
      //   constructors throw, so we just mark the whole method noexcept in order to avoid the
      //   code bloat to handle this case.
      next->arena = nullptr;
      T* ptr = reinterpret_cast<T*>(next.get()) - 1;
      ctor(*ptr, kj::mv(next), kj::fwd<Params>(params)...);
      ptr->arena = arena;
      KJ_IREQUIRE(reinterpret_cast<void*>(ptr) ==
                  reinterpret_cast<void*>(static_cast<PromiseArenaMember*>(ptr)),
          "PromiseArenaMember must be the leftmost inherited type.");
      return kj::Own<T, D>(ptr);
    }
  }
};

template <typename T, typename... Params>
static kj::Own<T, PromiseDisposer> allocPromise(Params&&... params) {
  // Allocate a PromiseNode without appending it to any existing promise arena. Space for a new
  // arena will be allocated.
  return PromiseDisposer::alloc<T>(kj::fwd<Params>(params)...);
}

template <typename T, bool arena = PromiseDisposer::canArenaAllocate<T>()>
struct FreePromiseNode;
template <typename T>
struct FreePromiseNode<T, true> {
  static inline void free(T* ptr) {
    // The object will have been allocated in an arena, so we only want to run the destructor.
    // The arena's memory will be freed separately.
    kj::dtor(*ptr);
  }
};
template <typename T>
struct FreePromiseNode<T, false> {
  static inline void free(T* ptr) {
    // The object will have been allocated separately on the heap.
    return delete ptr;
  }
};

template <typename T>
static void freePromise(T* ptr) {
  // Free a PromiseNode originally allocated using `allocPromise<T>()`. The implementation of
  // PromiseNode::destroy() must call this for any type that is allocated using allocPromise().
  FreePromiseNode<T>::free(ptr);
}

template <typename T, typename... Params>
static kj::Own<T, PromiseDisposer> appendPromise(OwnPromiseNode&& next, Params&&... params) {
  // Append a promise to the arena that currently ends with `next`. `next` is also still passed as
  // the first parameter to the new object's constructor.
  //
  // This is semantically the same as `allocPromise()` except that it may avoid the underlying
  // memory allocation. `next` must end up being destroyed before the new object (i.e. the new
  // object must never transfer away ownership of `next`).
  return PromiseDisposer::append<T>(kj::mv(next), kj::fwd<Params>(params)...);
}

// -------------------------------------------------------------------

inline ReadyNow::operator Promise<void>() const {
  return PromiseNode::to<Promise<void>>(readyNow());
}

template <typename T>
inline NeverDone::operator Promise<T>() const {
  return PromiseNode::to<Promise<T>>(neverDone());
}

// -------------------------------------------------------------------

class ImmediatePromiseNodeBase: public PromiseNode {
public:
  ImmediatePromiseNodeBase();
  ~ImmediatePromiseNodeBase() noexcept(false);

  void onReady(Event* event) noexcept override;
  void tracePromise(TraceBuilder& builder, bool stopAtNextEvent) override;
};

template <typename T>
class ImmediatePromiseNode final: public ImmediatePromiseNodeBase {
  // A promise that has already been resolved to an immediate value or exception.

public:
  ImmediatePromiseNode(ExceptionOr<T>&& result): result(kj::mv(result)) {}
  void destroy() override { freePromise(this); }

  void get(ExceptionOrValue& output) noexcept override {
    output.as<T>() = kj::mv(result);
  }

private:
  ExceptionOr<T> result;
};

class ImmediateBrokenPromiseNode final: public ImmediatePromiseNodeBase {
public:
  ImmediateBrokenPromiseNode(Exception&& exception);
  void destroy() override;

  void get(ExceptionOrValue& output) noexcept override;

private:
  Exception exception;
};

template <typename T, T value>
class ConstPromiseNode: public ImmediatePromiseNodeBase {
public:
  void destroy() override {}
  void get(ExceptionOrValue& output) noexcept override {
    output.as<T>() = value;
  }
};

// -------------------------------------------------------------------

class AttachmentPromiseNodeBase: public PromiseNode {
public:
  AttachmentPromiseNodeBase(OwnPromiseNode&& dependency);

  void onReady(Event* event) noexcept override;
  void get(ExceptionOrValue& output) noexcept override;
  void tracePromise(TraceBuilder& builder, bool stopAtNextEvent) override;

private:
  OwnPromiseNode dependency;

  void dropDependency();

  template <typename>
  friend class AttachmentPromiseNode;
};

template <typename Attachment>
class AttachmentPromiseNode final: public AttachmentPromiseNodeBase {
  // A PromiseNode that holds on to some object (usually, an Own<T>, but could be any movable
  // object) until the promise resolves.

public:
  AttachmentPromiseNode(OwnPromiseNode&& dependency, Attachment&& attachment)
      : AttachmentPromiseNodeBase(kj::mv(dependency)),
        attachment(kj::mv<Attachment>(attachment)) {}
  void destroy() override { freePromise(this); }

  ~AttachmentPromiseNode() noexcept(false) {
    // We need to make sure the dependency is deleted before we delete the attachment because the
    // dependency may be using the attachment.
    dropDependency();
  }

private:
  Attachment attachment;
};

// -------------------------------------------------------------------

#if __GNUC__ >= 8 && !__clang__
// GCC 8's class-memaccess warning rightly does not like the memcpy()'s below, but there's no
// "legal" way for us to extract the content of a PTMF so too bad.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wclass-memaccess"
#if __GNUC__ >= 11
// GCC 11's array-bounds is  similarly upset with us for digging into "private" implementation
// details. But the format is well-defined by the ABI which cannot change so please just let us
// do it kthx.
#pragma GCC diagnostic ignored "-Warray-bounds"
#endif
#endif

template <typename T, typename ReturnType, typename... ParamTypes>
void* getMethodStartAddress(T& obj, ReturnType (T::*method)(ParamTypes...));
template <typename T, typename ReturnType, typename... ParamTypes>
void* getMethodStartAddress(const T& obj, ReturnType (T::*method)(ParamTypes...) const);
// Given an object and a pointer-to-method, return the start address of the method's code. The
// intent is that this address can be used in a trace; addr2line should map it to the start of
// the function's definition. For virtual methods, this does a vtable lookup on `obj` to determine
// the address of the specific implementation (otherwise, `obj` wouldn't be needed).
//
// Note that if the method is overloaded or is a template, you will need to explicitly specify
// the param and return types, otherwise the compiler won't know which overload / template
// specialization you are requesting.

class PtmfHelper {
  // This class is a private helper for GetFunctorStartAddress and getMethodStartAddress(). The
  // class represents the internal representation of a pointer-to-member-function.

  template <typename... ParamTypes>
  friend struct GetFunctorStartAddress;
  template <typename T, typename ReturnType, typename... ParamTypes>
  friend void* getMethodStartAddress(T& obj, ReturnType (T::*method)(ParamTypes...));
  template <typename T, typename ReturnType, typename... ParamTypes>
  friend void* getMethodStartAddress(const T& obj, ReturnType (T::*method)(ParamTypes...) const);

#if __GNUG__

  void* ptr;
  ptrdiff_t adj;
  // Layout of a pointer-to-member-function used by GCC and compatible compilers.

  void* apply(const void* obj) {
#if defined(__arm__) || defined(__mips__) || defined(__aarch64__)
    if (adj & 1) {
      ptrdiff_t voff = (ptrdiff_t)ptr;
#else
    ptrdiff_t voff = (ptrdiff_t)ptr;
    if (voff & 1) {
      voff &= ~1;
#endif
      return *(void**)(*(char**)obj + voff);
    } else {
      return ptr;
    }
  }

#define BODY \
    PtmfHelper result; \
    static_assert(sizeof(p) == sizeof(result), "unknown ptmf layout"); \
    memcpy(&result, &p, sizeof(result)); \
    return result

#else  // __GNUG__

  void* apply(const void* obj) { return nullptr; }
  // TODO(port):  PTMF instruction address extraction

#define BODY return PtmfHelper{}

#endif  // __GNUG__, else

  template <typename R, typename C, typename... P, typename F>
  static PtmfHelper from(F p) { BODY; }
  // Create a PtmfHelper from some arbitrary pointer-to-member-function which is not
  // overloaded nor a template. In this case the compiler is able to deduce the full function
  // signature directly given the name since there is only one function with that name.

  template <typename R, typename C, typename... P>
  static PtmfHelper from(R (C::*p)(NoInfer<P>...)) { BODY; }
  template <typename R, typename C, typename... P>
  static PtmfHelper from(R (C::*p)(NoInfer<P>...) const) { BODY; }
  // Create a PtmfHelper from some poniter-to-member-function which is a template. In this case
  // the function must match exactly the containing type C, return type R, and parameter types P...
  // GetFunctorStartAddress normally specifies exactly the correct C and R, but can only make a
  // guess at P. Luckily, if the function parameters are template parameters then it's not
  // necessary to be precise about P.
#undef BODY
};

#if __GNUC__ >= 8 && !__clang__
#pragma GCC diagnostic pop
#endif

template <typename T, typename ReturnType, typename... ParamTypes>
void* getMethodStartAddress(T& obj, ReturnType (T::*method)(ParamTypes...)) {
  return PtmfHelper::from<ReturnType, T, ParamTypes...>(method).apply(&obj);
}
template <typename T, typename ReturnType, typename... ParamTypes>
void* getMethodStartAddress(const T& obj, ReturnType (T::*method)(ParamTypes...) const) {
  return PtmfHelper::from<ReturnType, T, ParamTypes...>(method).apply(&obj);
}

template <typename... ParamTypes>
struct GetFunctorStartAddress {
  // Given a functor (any object defining operator()), return the start address of the function,
  // suitable for passing to addr2line to obtain a source file/line for debugging purposes.
  //
  // This turns out to be incredibly hard to implement in the presence of overloaded or templated
  // functors. Therefore, we impose these specific restrictions, specific to our use case:
  // - Overloading is not allowed, but templating is. (Generally we only intend to support lambdas
  //   anyway.)
  // - The template parameters to GetFunctorStartAddress specify a hint as to the expected
  //   parameter types. If the functor is templated, its parameters must match exactly these types.
  //   (If it's not templated, ParamTypes are ignored.)

  template <typename Func>
  static void* apply(Func&& func) {
    typedef decltype(func(instance<ParamTypes>()...)) ReturnType;
    return PtmfHelper::from<ReturnType, Decay<Func>, ParamTypes...>(
        &Decay<Func>::operator()).apply(&func);
  }
};

template <>
struct GetFunctorStartAddress<Void&&>: public GetFunctorStartAddress<> {};
// Hack for TransformPromiseNode use case: an input type of `Void` indicates that the function
// actually has no parameters.

class TransformPromiseNodeBase: public PromiseNode {
public:
  TransformPromiseNodeBase(OwnPromiseNode&& dependency, void* continuationTracePtr);

  void onReady(Event* event) noexcept override;
  void get(ExceptionOrValue& output) noexcept override;
  void tracePromise(TraceBuilder& builder, bool stopAtNextEvent) override;

private:
  OwnPromiseNode dependency;
  void* continuationTracePtr;

  void dropDependency();
  void getDepResult(ExceptionOrValue& output);

  virtual void getImpl(ExceptionOrValue& output) = 0;

  template <typename, typename, typename, typename>
  friend class TransformPromiseNode;
};

template <typename T, typename DepT, typename Func, typename ErrorFunc>
class TransformPromiseNode final: public TransformPromiseNodeBase {
  // A PromiseNode that transforms the result of another PromiseNode through an application-provided
  // function (implements `then()`).

public:
  TransformPromiseNode(OwnPromiseNode&& dependency, Func&& func, ErrorFunc&& errorHandler,
                       void* continuationTracePtr)
      : TransformPromiseNodeBase(kj::mv(dependency), continuationTracePtr),
        func(kj::fwd<Func>(func)), errorHandler(kj::fwd<ErrorFunc>(errorHandler)) {}
  void destroy() override { freePromise(this); }

  ~TransformPromiseNode() noexcept(false) {
    // We need to make sure the dependency is deleted before we delete the continuations because it
    // is a common pattern for the continuations to hold ownership of objects that might be in-use
    // by the dependency.
    dropDependency();
  }

private:
  Func func;
  ErrorFunc errorHandler;

  void getImpl(ExceptionOrValue& output) override {
    ExceptionOr<DepT> depResult;
    getDepResult(depResult);
    KJ_IF_MAYBE(depException, depResult.exception) {
      output.as<T>() = handle(
          MaybeVoidCaller<Exception, FixVoid<ReturnType<ErrorFunc, Exception>>>::apply(
              errorHandler, kj::mv(*depException)));
    } else KJ_IF_MAYBE(depValue, depResult.value) {
      output.as<T>() = handle(MaybeVoidCaller<DepT, T>::apply(func, kj::mv(*depValue)));
    }
  }

  ExceptionOr<T> handle(T&& value) {
    return kj::mv(value);
  }
  ExceptionOr<T> handle(PropagateException::Bottom&& value) {
    return ExceptionOr<T>(false, value.asException());
  }
};

// -------------------------------------------------------------------

class ForkHubBase;
using OwnForkHubBase = Own<ForkHubBase, ForkHubBase>;

class ForkBranchBase: public PromiseNode {
public:
  ForkBranchBase(OwnForkHubBase&& hub);
  ~ForkBranchBase() noexcept(false);

  void hubReady() noexcept;
  // Called by the hub to indicate that it is ready.

  // implements PromiseNode ------------------------------------------
  void onReady(Event* event) noexcept override;
  void tracePromise(TraceBuilder& builder, bool stopAtNextEvent) override;

protected:
  inline ExceptionOrValue& getHubResultRef();

  void releaseHub(ExceptionOrValue& output);
  // Release the hub.  If an exception is thrown, add it to `output`.

private:
  OnReadyEvent onReadyEvent;

  OwnForkHubBase hub;
  ForkBranchBase* next = nullptr;
  ForkBranchBase** prevPtr = nullptr;

  friend class ForkHubBase;
};

template <typename T> T copyOrAddRef(T& t) { return t; }
template <typename T> Own<T> copyOrAddRef(Own<T>& t) { return t->addRef(); }
template <typename T> Maybe<Own<T>> copyOrAddRef(Maybe<Own<T>>& t) {
  return t.map([](Own<T>& ptr) {
    return ptr->addRef();
  });
}

template <typename T>
class ForkBranch final: public ForkBranchBase {
  // A PromiseNode that implements one branch of a fork -- i.e. one of the branches that receives
  // a const reference.

public:
  ForkBranch(OwnForkHubBase&& hub): ForkBranchBase(kj::mv(hub)) {}
  void destroy() override { freePromise(this); }

  void get(ExceptionOrValue& output) noexcept override {
    ExceptionOr<T>& hubResult = getHubResultRef().template as<T>();
    KJ_IF_MAYBE(value, hubResult.value) {
      output.as<T>().value = copyOrAddRef(*value);
    } else {
      output.as<T>().value = nullptr;
    }
    output.exception = hubResult.exception;
    releaseHub(output);
  }
};

template <typename T, size_t index>
class SplitBranch final: public ForkBranchBase {
  // A PromiseNode that implements one branch of a fork -- i.e. one of the branches that receives
  // a const reference.

public:
  SplitBranch(OwnForkHubBase&& hub): ForkBranchBase(kj::mv(hub)) {}
  void destroy() override { freePromise(this); }

  typedef kj::Decay<decltype(kj::get<index>(kj::instance<T>()))> Element;

  void get(ExceptionOrValue& output) noexcept override {
    ExceptionOr<T>& hubResult = getHubResultRef().template as<T>();
    KJ_IF_MAYBE(value, hubResult.value) {
      output.as<Element>().value = kj::mv(kj::get<index>(*value));
    } else {
      output.as<Element>().value = nullptr;
    }
    output.exception = hubResult.exception;
    releaseHub(output);
  }
};

// -------------------------------------------------------------------

class ForkHubBase: public PromiseArenaMember, protected Event {
public:
  ForkHubBase(OwnPromiseNode&& inner, ExceptionOrValue& resultRef, SourceLocation location);

  inline ExceptionOrValue& getResultRef() { return resultRef; }

  inline bool isShared() const { return refcount > 1; }

  Own<ForkHubBase, ForkHubBase> addRef() {
    ++refcount;
    return Own<ForkHubBase, ForkHubBase>(this);
  }

  static void dispose(ForkHubBase* obj) {
    if (--obj->refcount == 0) {
      PromiseDisposer::dispose(obj);
    }
  }

private:
  uint refcount = 1;
  // We manually implement refcounting for ForkHubBase so that we can use it together with
  // PromiseDisposer's arena allocation.

  OwnPromiseNode inner;
  ExceptionOrValue& resultRef;

  ForkBranchBase* headBranch = nullptr;
  ForkBranchBase** tailBranch = &headBranch;
  // Tail becomes null once the inner promise is ready and all branches have been notified.

  Maybe<Own<Event>> fire() override;
  void traceEvent(TraceBuilder& builder) override;

  friend class ForkBranchBase;
};

template <typename T>
class ForkHub final: public ForkHubBase {
  // A PromiseNode that implements the hub of a fork.  The first call to Promise::fork() replaces
  // the promise's outer node with a ForkHub, and subsequent calls add branches to that hub (if
  // possible).

public:
  ForkHub(OwnPromiseNode&& inner, SourceLocation location)
      : ForkHubBase(kj::mv(inner), result, location) {}
  void destroy() override { freePromise(this); }

  Promise<_::UnfixVoid<T>> addBranch() {
    return _::PromiseNode::to<Promise<_::UnfixVoid<T>>>(
        allocPromise<ForkBranch<T>>(addRef()));
  }

  _::SplitTuplePromise<T> split(SourceLocation location) {
    return splitImpl(MakeIndexes<tupleSize<T>()>(), location);
  }

private:
  ExceptionOr<T> result;

  template <size_t... indexes>
  _::SplitTuplePromise<T> splitImpl(Indexes<indexes...>, SourceLocation location) {
    return kj::tuple(addSplit<indexes>(location)...);
  }

  template <size_t index>
  ReducePromises<typename SplitBranch<T, index>::Element> addSplit(SourceLocation location) {
    return _::PromiseNode::to<ReducePromises<typename SplitBranch<T, index>::Element>>(
        maybeChain(allocPromise<SplitBranch<T, index>>(addRef()),
                   implicitCast<typename SplitBranch<T, index>::Element*>(nullptr),
                   location));
  }
};

inline ExceptionOrValue& ForkBranchBase::getHubResultRef() {
  return hub->getResultRef();
}

// -------------------------------------------------------------------

class ChainPromiseNode final: public PromiseNode, public Event {
  // Promise node which reduces Promise<Promise<T>> to Promise<T>.
  //
  // `Event` is only a public base class because otherwise we can't cast Own<ChainPromiseNode> to
  // Own<Event>.  Ugh, templates and private...

public:
  explicit ChainPromiseNode(OwnPromiseNode inner, SourceLocation location);
  ~ChainPromiseNode() noexcept(false);
  void destroy() override;

  void onReady(Event* event) noexcept override;
  void setSelfPointer(OwnPromiseNode* selfPtr) noexcept override;
  void get(ExceptionOrValue& output) noexcept override;
  void tracePromise(TraceBuilder& builder, bool stopAtNextEvent) override;

private:
  enum State {
    STEP1,
    STEP2
  };

  State state;

  OwnPromiseNode inner;
  // In STEP1, a PromiseNode for a Promise<T>.
  // In STEP2, a PromiseNode for a T.

  Event* onReadyEvent = nullptr;
  OwnPromiseNode* selfPtr = nullptr;

  Maybe<Own<Event>> fire() override;
  void traceEvent(TraceBuilder& builder) override;
};

template <typename T>
OwnPromiseNode maybeChain(OwnPromiseNode&& node, Promise<T>*, SourceLocation location) {
  return appendPromise<ChainPromiseNode>(kj::mv(node), location);
}

template <typename T>
OwnPromiseNode&& maybeChain(OwnPromiseNode&& node, T*, SourceLocation location) {
  return kj::mv(node);
}

template <typename T, typename Result = decltype(T::reducePromise(instance<Promise<T>>()))>
inline Result maybeReduce(Promise<T>&& promise, bool) {
  return T::reducePromise(kj::mv(promise));
}

template <typename T>
inline Promise<T> maybeReduce(Promise<T>&& promise, ...) {
  return kj::mv(promise);
}

// -------------------------------------------------------------------

class ExclusiveJoinPromiseNode final: public PromiseNode {
public:
  ExclusiveJoinPromiseNode(OwnPromiseNode left, OwnPromiseNode right, SourceLocation location);
  ~ExclusiveJoinPromiseNode() noexcept(false);
  void destroy() override;

  void onReady(Event* event) noexcept override;
  void get(ExceptionOrValue& output) noexcept override;
  void tracePromise(TraceBuilder& builder, bool stopAtNextEvent) override;

private:
  class Branch: public Event {
  public:
    Branch(ExclusiveJoinPromiseNode& joinNode, OwnPromiseNode dependency,
           SourceLocation location);
    ~Branch() noexcept(false);

    bool get(ExceptionOrValue& output);
    // Returns true if this is the side that finished.

    Maybe<Own<Event>> fire() override;
    void traceEvent(TraceBuilder& builder) override;

  private:
    ExclusiveJoinPromiseNode& joinNode;
    OwnPromiseNode dependency;

    friend class ExclusiveJoinPromiseNode;
  };

  Branch left;
  Branch right;
  OnReadyEvent onReadyEvent;
};

// -------------------------------------------------------------------

enum class ArrayJoinBehavior {
  LAZY,
  EAGER,
};

class ArrayJoinPromiseNodeBase: public PromiseNode {
public:
  ArrayJoinPromiseNodeBase(Array<OwnPromiseNode> promises,
                           ExceptionOrValue* resultParts, size_t partSize,
                           SourceLocation location,
                           ArrayJoinBehavior joinBehavior);
  ~ArrayJoinPromiseNodeBase() noexcept(false);

  void onReady(Event* event) noexcept override final;
  void get(ExceptionOrValue& output) noexcept override final;
  void tracePromise(TraceBuilder& builder, bool stopAtNextEvent) override final;

protected:
  virtual void getNoError(ExceptionOrValue& output) noexcept = 0;
  // Called to compile the result only in the case where there were no errors.

private:
  const ArrayJoinBehavior joinBehavior;

  uint countLeft;
  OnReadyEvent onReadyEvent;
  bool armed = false;

  class Branch final: public Event {
  public:
    Branch(ArrayJoinPromiseNodeBase& joinNode, OwnPromiseNode dependency,
           ExceptionOrValue& output, SourceLocation location);
    ~Branch() noexcept(false);

    Maybe<Own<Event>> fire() override;
    void traceEvent(TraceBuilder& builder) override;

  private:
    ArrayJoinPromiseNodeBase& joinNode;
    OwnPromiseNode dependency;
    ExceptionOrValue& output;

    friend class ArrayJoinPromiseNodeBase;
  };

  Array<Branch> branches;
};

template <typename T>
class ArrayJoinPromiseNode final: public ArrayJoinPromiseNodeBase {
public:
  ArrayJoinPromiseNode(Array<OwnPromiseNode> promises,
                       Array<ExceptionOr<T>> resultParts,
                       SourceLocation location,
                       ArrayJoinBehavior joinBehavior)
      : ArrayJoinPromiseNodeBase(kj::mv(promises), resultParts.begin(), sizeof(ExceptionOr<T>),
                                 location, joinBehavior),
        resultParts(kj::mv(resultParts)) {}
  void destroy() override { freePromise(this); }

protected:
  void getNoError(ExceptionOrValue& output) noexcept override {
    auto builder = heapArrayBuilder<T>(resultParts.size());
    for (auto& part: resultParts) {
      KJ_IASSERT(part.value != nullptr,
                 "Bug in KJ promise framework:  Promise result had neither value no exception.");
      builder.add(kj::mv(*_::readMaybe(part.value)));
    }
    output.as<Array<T>>() = builder.finish();
  }

private:
  Array<ExceptionOr<T>> resultParts;
};

template <>
class ArrayJoinPromiseNode<void> final: public ArrayJoinPromiseNodeBase {
public:
  ArrayJoinPromiseNode(Array<OwnPromiseNode> promises,
                       Array<ExceptionOr<_::Void>> resultParts,
                       SourceLocation location,
                       ArrayJoinBehavior joinBehavior);
  ~ArrayJoinPromiseNode();
  void destroy() override;

protected:
  void getNoError(ExceptionOrValue& output) noexcept override;

private:
  Array<ExceptionOr<_::Void>> resultParts;
};

// -------------------------------------------------------------------

class EagerPromiseNodeBase: public PromiseNode, protected Event {
  // A PromiseNode that eagerly evaluates its dependency even if its dependent does not eagerly
  // evaluate it.

public:
  EagerPromiseNodeBase(OwnPromiseNode&& dependency, ExceptionOrValue& resultRef,
                       SourceLocation location);

  void onReady(Event* event) noexcept override;
  void tracePromise(TraceBuilder& builder, bool stopAtNextEvent) override;

private:
  OwnPromiseNode dependency;
  OnReadyEvent onReadyEvent;

  ExceptionOrValue& resultRef;

  Maybe<Own<Event>> fire() override;
  void traceEvent(TraceBuilder& builder) override;
};

template <typename T>
class EagerPromiseNode final: public EagerPromiseNodeBase {
public:
  EagerPromiseNode(OwnPromiseNode&& dependency, SourceLocation location)
      : EagerPromiseNodeBase(kj::mv(dependency), result, location) {}
  void destroy() override { freePromise(this); }

  void get(ExceptionOrValue& output) noexcept override {
    output.as<T>() = kj::mv(result);
  }

private:
  ExceptionOr<T> result;
};

template <typename T>
OwnPromiseNode spark(OwnPromiseNode&& node, SourceLocation location) {
  // Forces evaluation of the given node to begin as soon as possible, even if no one is waiting
  // on it.
  return appendPromise<EagerPromiseNode<T>>(kj::mv(node), location);
}

// -------------------------------------------------------------------

class AdapterPromiseNodeBase: public PromiseNode {
public:
  void onReady(Event* event) noexcept override;
  void tracePromise(TraceBuilder& builder, bool stopAtNextEvent) override;

protected:
  inline void setReady() {
    onReadyEvent.arm();
  }

private:
  OnReadyEvent onReadyEvent;
};

template <typename T, typename Adapter>
class AdapterPromiseNode final: public AdapterPromiseNodeBase,
                                private PromiseFulfiller<UnfixVoid<T>> {
  // A PromiseNode that wraps a PromiseAdapter.

public:
  template <typename... Params>
  AdapterPromiseNode(Params&&... params)
      : adapter(static_cast<PromiseFulfiller<UnfixVoid<T>>&>(*this), kj::fwd<Params>(params)...) {}
  void destroy() override { freePromise(this); }

  void get(ExceptionOrValue& output) noexcept override {
    KJ_IREQUIRE(!isWaiting());
    output.as<T>() = kj::mv(result);
  }

private:
  ExceptionOr<T> result;
  bool waiting = true;
  Adapter adapter;

  void fulfill(T&& value) override {
    if (waiting) {
      waiting = false;
      result = ExceptionOr<T>(kj::mv(value));
      setReady();
    }
  }

  void reject(Exception&& exception) override {
    if (waiting) {
      waiting = false;
      result = ExceptionOr<T>(false, kj::mv(exception));
      setReady();
    }
  }

  bool isWaiting() override {
    return waiting;
  }
};

// -------------------------------------------------------------------

class FiberBase: public PromiseNode, private Event {
  // Base class for the outer PromiseNode representing a fiber.

public:
  explicit FiberBase(size_t stackSize, _::ExceptionOrValue& result, SourceLocation location);
  explicit FiberBase(const FiberPool& pool, _::ExceptionOrValue& result, SourceLocation location);
  ~FiberBase() noexcept(false);

  void start() { armDepthFirst(); }
  // Call immediately after construction to begin executing the fiber.

  class WaitDoneEvent;

  void onReady(_::Event* event) noexcept override;
  void tracePromise(TraceBuilder& builder, bool stopAtNextEvent) override;

protected:
  bool isFinished() { return state == FINISHED; }
  void cancel();

private:
  enum { WAITING, RUNNING, CANCELED, FINISHED } state;

  _::PromiseNode* currentInner = nullptr;
  OnReadyEvent onReadyEvent;
  Own<FiberStack> stack;
  _::ExceptionOrValue& result;

  void run();
  virtual void runImpl(WaitScope& waitScope) = 0;

  Maybe<Own<Event>> fire() override;
  void traceEvent(TraceBuilder& builder) override;
  // Implements Event. Each time the event is fired, switchToFiber() is called.

  friend class FiberStack;
  friend void _::waitImpl(_::OwnPromiseNode&& node, _::ExceptionOrValue& result,
                          WaitScope& waitScope, SourceLocation location);
  friend bool _::pollImpl(_::PromiseNode& node, WaitScope& waitScope, SourceLocation location);
};

template <typename Func>
class Fiber final: public FiberBase {
public:
  explicit Fiber(size_t stackSize, Func&& func, SourceLocation location)
      : FiberBase(stackSize, result, location), func(kj::fwd<Func>(func)) {}
  explicit Fiber(const FiberPool& pool, Func&& func, SourceLocation location)
      : FiberBase(pool, result, location), func(kj::fwd<Func>(func)) {}
  ~Fiber() noexcept(false) { cancel(); }
  void destroy() override { freePromise(this); }

  typedef FixVoid<decltype(kj::instance<Func&>()(kj::instance<WaitScope&>()))> ResultType;

  void get(ExceptionOrValue& output) noexcept override {
    KJ_IREQUIRE(isFinished());
    output.as<ResultType>() = kj::mv(result);
  }

private:
  Func func;
  ExceptionOr<ResultType> result;

  void runImpl(WaitScope& waitScope) override {
    result.template as<ResultType>() =
        MaybeVoidCaller<WaitScope&, ResultType>::apply(func, waitScope);
  }
};

}  // namespace _ (private)

// =======================================================================================

template <typename T>
Promise<T>::Promise(_::FixVoid<T> value)
    : PromiseBase(_::allocPromise<_::ImmediatePromiseNode<_::FixVoid<T>>>(kj::mv(value))) {}

template <typename T>
Promise<T>::Promise(kj::Exception&& exception)
    : PromiseBase(_::allocPromise<_::ImmediateBrokenPromiseNode>(kj::mv(exception))) {}

template <typename T>
template <typename Func, typename ErrorFunc>
PromiseForResult<Func, T> Promise<T>::then(Func&& func, ErrorFunc&& errorHandler,
                                           SourceLocation location) {
  typedef _::FixVoid<_::ReturnType<Func, T>> ResultT;

  void* continuationTracePtr = _::GetFunctorStartAddress<_::FixVoid<T>&&>::apply(func);
  _::OwnPromiseNode intermediate =
      _::appendPromise<_::TransformPromiseNode<ResultT, _::FixVoid<T>, Func, ErrorFunc>>(
          kj::mv(node), kj::fwd<Func>(func), kj::fwd<ErrorFunc>(errorHandler),
          continuationTracePtr);
  auto result = _::PromiseNode::to<_::ChainPromises<_::ReturnType<Func, T>>>(
      _::maybeChain(kj::mv(intermediate), implicitCast<ResultT*>(nullptr), location));
  return _::maybeReduce(kj::mv(result), false);
}

namespace _ {  // private

template <typename T>
struct IdentityFunc {
  inline T operator()(T&& value) const {
    return kj::mv(value);
  }
};
template <typename T>
struct IdentityFunc<Promise<T>> {
  inline Promise<T> operator()(T&& value) const {
    return kj::mv(value);
  }
};
template <>
struct IdentityFunc<void> {
  inline void operator()() const {}
};
template <>
struct IdentityFunc<Promise<void>> {
  Promise<void> operator()() const;
  // This can't be inline because it will make the translation unit depend on kj-async. Awkwardly,
  // Cap'n Proto relies on being able to include this header without creating such a link-time
  // dependency.
};

}  // namespace _ (private)

template <typename T>
template <typename ErrorFunc>
Promise<T> Promise<T>::catch_(ErrorFunc&& errorHandler, SourceLocation location) {
  // then()'s ErrorFunc can only return a Promise if Func also returns a Promise. In this case,
  // Func is being filled in automatically. We want to make sure ErrorFunc can return a Promise,
  // but we don't want the extra overhead of promise chaining if ErrorFunc doesn't actually
  // return a promise. So we make our Func return match ErrorFunc.
  typedef _::IdentityFunc<decltype(errorHandler(instance<Exception&&>()))> Func;
  typedef _::FixVoid<_::ReturnType<Func, T>> ResultT;

  // The reason catch_() isn't simply implemented in terms of then() is because we want the trace
  // pointer to be based on ErrorFunc rather than Func.
  void* continuationTracePtr = _::GetFunctorStartAddress<kj::Exception&&>::apply(errorHandler);
  _::OwnPromiseNode intermediate =
      _::appendPromise<_::TransformPromiseNode<ResultT, _::FixVoid<T>, Func, ErrorFunc>>(
          kj::mv(node), Func(), kj::fwd<ErrorFunc>(errorHandler), continuationTracePtr);
  auto result = _::PromiseNode::to<_::ChainPromises<_::ReturnType<Func, T>>>(
      _::maybeChain(kj::mv(intermediate), implicitCast<ResultT*>(nullptr), location));
  return _::maybeReduce(kj::mv(result), false);
}

template <typename T>
T Promise<T>::wait(WaitScope& waitScope, SourceLocation location) {
  _::ExceptionOr<_::FixVoid<T>> result;
  _::waitImpl(kj::mv(node), result, waitScope, location);
  return convertToReturn(kj::mv(result));
}

template <typename T>
bool Promise<T>::poll(WaitScope& waitScope, SourceLocation location) {
  return _::pollImpl(*node, waitScope, location);
}

template <typename T>
ForkedPromise<T> Promise<T>::fork(SourceLocation location) {
  return ForkedPromise<T>(false,
      _::PromiseDisposer::alloc<_::ForkHub<_::FixVoid<T>>, _::ForkHubBase>(kj::mv(node), location));
}

template <typename T>
Promise<T> ForkedPromise<T>::addBranch() {
  return hub->addBranch();
}

template <typename T>
bool ForkedPromise<T>::hasBranches() {
  return hub->isShared();
}

template <typename T>
_::SplitTuplePromise<T> Promise<T>::split(SourceLocation location) {
  return _::PromiseDisposer::alloc<_::ForkHub<_::FixVoid<T>>, _::ForkHubBase>(
      kj::mv(node), location)->split(location);
}

template <typename T>
Promise<T> Promise<T>::exclusiveJoin(Promise<T>&& other, SourceLocation location) {
  return Promise(false, _::appendPromise<_::ExclusiveJoinPromiseNode>(
      kj::mv(node), kj::mv(other.node), location));
}

template <typename T>
template <typename... Attachments>
Promise<T> Promise<T>::attach(Attachments&&... attachments) {
  return Promise(false, _::appendPromise<_::AttachmentPromiseNode<Tuple<Attachments...>>>(
      kj::mv(node), kj::tuple(kj::fwd<Attachments>(attachments)...)));
}

template <typename T>
template <typename ErrorFunc>
Promise<T> Promise<T>::eagerlyEvaluate(ErrorFunc&& errorHandler, SourceLocation location) {
  // See catch_() for commentary.
  return Promise(false, _::spark<_::FixVoid<T>>(then(
      _::IdentityFunc<decltype(errorHandler(instance<Exception&&>()))>(),
      kj::fwd<ErrorFunc>(errorHandler)).node, location));
}

template <typename T>
Promise<T> Promise<T>::eagerlyEvaluate(decltype(nullptr), SourceLocation location) {
  return Promise(false, _::spark<_::FixVoid<T>>(kj::mv(node), location));
}

template <typename T>
kj::String Promise<T>::trace() {
  return PromiseBase::trace();
}

template <typename T, T value>
inline Promise<T> constPromise() {
  static _::ConstPromiseNode<T, value> NODE;
  return _::PromiseNode::to<Promise<T>>(_::OwnPromiseNode(&NODE));
}

template <typename Func>
inline PromiseForResult<Func, void> evalLater(Func&& func) {
  return _::yield().then(kj::fwd<Func>(func), _::PropagateException());
}

template <typename Func>
inline PromiseForResult<Func, void> evalLast(Func&& func) {
  return _::yieldHarder().then(kj::fwd<Func>(func), _::PropagateException());
}

template <typename Func>
inline PromiseForResult<Func, void> evalNow(Func&& func) {
  PromiseForResult<Func, void> result = nullptr;
  KJ_IF_MAYBE(e, kj::runCatchingExceptions([&]() {
    result = func();
  })) {
    result = kj::mv(*e);
  }
  return result;
}

template <typename Func>
struct RetryOnDisconnect_ {
  static inline PromiseForResult<Func, void> apply(Func&& func) {
    return evalLater([func = kj::mv(func)]() mutable -> PromiseForResult<Func, void> {
      auto promise = evalNow(func);
      return promise.catch_([func = kj::mv(func)](kj::Exception&& e) mutable -> PromiseForResult<Func, void> {
        if (e.getType() == kj::Exception::Type::DISCONNECTED) {
          return func();
        } else {
          return kj::mv(e);
        }
      });
    });
  }
};
template <typename Func>
struct RetryOnDisconnect_<Func&> {
  // Specialization for references. Needed because the syntax for capturing references in a
  // lambda is different. :(
  static inline PromiseForResult<Func, void> apply(Func& func) {
    auto promise = evalLater(func);
    return promise.catch_([&func](kj::Exception&& e) -> PromiseForResult<Func, void> {
      if (e.getType() == kj::Exception::Type::DISCONNECTED) {
        return func();
      } else {
        return kj::mv(e);
      }
    });
  }
};

template <typename Func>
inline PromiseForResult<Func, void> retryOnDisconnect(Func&& func) {
  return RetryOnDisconnect_<Func>::apply(kj::fwd<Func>(func));
}

template <typename Func>
inline PromiseForResult<Func, WaitScope&> startFiber(
    size_t stackSize, Func&& func, SourceLocation location) {
  typedef _::FixVoid<_::ReturnType<Func, WaitScope&>> ResultT;

  auto intermediate = _::allocPromise<_::Fiber<Func>>(
      stackSize, kj::fwd<Func>(func), location);
  intermediate->start();
  auto result = _::PromiseNode::to<_::ChainPromises<_::ReturnType<Func, WaitScope&>>>(
      _::maybeChain(kj::mv(intermediate), implicitCast<ResultT*>(nullptr), location));
  return _::maybeReduce(kj::mv(result), false);
}

template <typename Func>
inline PromiseForResult<Func, WaitScope&> FiberPool::startFiber(
    Func&& func, SourceLocation location) const {
  typedef _::FixVoid<_::ReturnType<Func, WaitScope&>> ResultT;

  auto intermediate = _::allocPromise<_::Fiber<Func>>(
      *this, kj::fwd<Func>(func), location);
  intermediate->start();
  auto result = _::PromiseNode::to<_::ChainPromises<_::ReturnType<Func, WaitScope&>>>(
      _::maybeChain(kj::mv(intermediate), implicitCast<ResultT*>(nullptr), location));
  return _::maybeReduce(kj::mv(result), false);
}

template <typename T>
template <typename ErrorFunc>
void Promise<T>::detach(ErrorFunc&& errorHandler) {
  return _::detach(then([](T&&) {}, kj::fwd<ErrorFunc>(errorHandler)));
}

template <>
template <typename ErrorFunc>
void Promise<void>::detach(ErrorFunc&& errorHandler) {
  return _::detach(then([]() {}, kj::fwd<ErrorFunc>(errorHandler)));
}

template <typename T>
Promise<Array<T>> joinPromises(Array<Promise<T>>&& promises, SourceLocation location) {
  return _::PromiseNode::to<Promise<Array<T>>>(_::allocPromise<_::ArrayJoinPromiseNode<T>>(
      KJ_MAP(p, promises) { return _::PromiseNode::from(kj::mv(p)); },
      heapArray<_::ExceptionOr<T>>(promises.size()), location,
      _::ArrayJoinBehavior::LAZY));
}

template <typename T>
Promise<Array<T>> joinPromisesFailFast(Array<Promise<T>>&& promises, SourceLocation location) {
  return _::PromiseNode::to<Promise<Array<T>>>(_::allocPromise<_::ArrayJoinPromiseNode<T>>(
      KJ_MAP(p, promises) { return _::PromiseNode::from(kj::mv(p)); },
      heapArray<_::ExceptionOr<T>>(promises.size()), location,
      _::ArrayJoinBehavior::EAGER));
}

// =======================================================================================

namespace _ {  // private

class WeakFulfillerBase: protected kj::Disposer {
protected:
  WeakFulfillerBase(): inner(nullptr) {}
  virtual ~WeakFulfillerBase() noexcept(false) {}

  template <typename T>
  inline PromiseFulfiller<T>* getInner() {
    return static_cast<PromiseFulfiller<T>*>(inner);
  };
  template <typename T>
  inline void setInner(PromiseFulfiller<T>* ptr) {
    inner = ptr;
  };

private:
  mutable PromiseRejector* inner;

  void disposeImpl(void* pointer) const override;
};

template <typename T>
class WeakFulfiller final: public PromiseFulfiller<T>, public WeakFulfillerBase {
  // A wrapper around PromiseFulfiller which can be detached.
  //
  // There are a couple non-trivialities here:
  // - If the WeakFulfiller is discarded, we want the promise it fulfills to be implicitly
  //   rejected.
  // - We cannot destroy the WeakFulfiller until the application has discarded it *and* it has been
  //   detached from the underlying fulfiller, because otherwise the later detach() call will go
  //   to a dangling pointer.  Essentially, WeakFulfiller is reference counted, although the
  //   refcount never goes over 2 and we manually implement the refcounting because we need to do
  //   other special things when each side detaches anyway.  To this end, WeakFulfiller is its own
  //   Disposer -- dispose() is called when the application discards its owned pointer to the
  //   fulfiller and detach() is called when the promise is destroyed.

public:
  KJ_DISALLOW_COPY_AND_MOVE(WeakFulfiller);

  static kj::Own<WeakFulfiller> make() {
    WeakFulfiller* ptr = new WeakFulfiller;
    return Own<WeakFulfiller>(ptr, *ptr);
  }

  void fulfill(FixVoid<T>&& value) override {
    if (getInner<T>() != nullptr) {
      getInner<T>()->fulfill(kj::mv(value));
    }
  }

  void reject(Exception&& exception) override {
    if (getInner<T>() != nullptr) {
      getInner<T>()->reject(kj::mv(exception));
    }
  }

  bool isWaiting() override {
    return getInner<T>() != nullptr && getInner<T>()->isWaiting();
  }

  void attach(PromiseFulfiller<T>& newInner) {
    setInner<T>(&newInner);
  }

  void detach(PromiseFulfiller<T>& from) {
    if (getInner<T>() == nullptr) {
      // Already disposed.
      delete this;
    } else {
      KJ_IREQUIRE(getInner<T>() == &from);
      setInner<T>(nullptr);
    }
  }

private:
  WeakFulfiller() {}
};

template <typename T>
class PromiseAndFulfillerAdapter {
public:
  PromiseAndFulfillerAdapter(PromiseFulfiller<T>& fulfiller,
                             WeakFulfiller<T>& wrapper)
      : fulfiller(fulfiller), wrapper(wrapper) {
    wrapper.attach(fulfiller);
  }

  ~PromiseAndFulfillerAdapter() noexcept(false) {
    wrapper.detach(fulfiller);
  }

private:
  PromiseFulfiller<T>& fulfiller;
  WeakFulfiller<T>& wrapper;
};

}  // namespace _ (private)

template <typename T>
template <typename Func>
bool PromiseFulfiller<T>::rejectIfThrows(Func&& func) {
  KJ_IF_MAYBE(exception, kj::runCatchingExceptions(kj::mv(func))) {
    reject(kj::mv(*exception));
    return false;
  } else {
    return true;
  }
}

template <typename Func>
bool PromiseFulfiller<void>::rejectIfThrows(Func&& func) {
  KJ_IF_MAYBE(exception, kj::runCatchingExceptions(kj::mv(func))) {
    reject(kj::mv(*exception));
    return false;
  } else {
    return true;
  }
}

template <typename T, typename Adapter, typename... Params>
_::ReducePromises<T> newAdaptedPromise(Params&&... adapterConstructorParams) {
  _::OwnPromiseNode intermediate(
      _::allocPromise<_::AdapterPromiseNode<_::FixVoid<T>, Adapter>>(
          kj::fwd<Params>(adapterConstructorParams)...));
  // We can't capture SourceLocation in this function's arguments since it is a vararg template. :(
  return _::PromiseNode::to<_::ReducePromises<T>>(
      _::maybeChain(kj::mv(intermediate), implicitCast<T*>(nullptr), SourceLocation()));
}

template <typename T>
PromiseFulfillerPair<T> newPromiseAndFulfiller(SourceLocation location) {
  auto wrapper = _::WeakFulfiller<T>::make();

  _::OwnPromiseNode intermediate(
      _::allocPromise<_::AdapterPromiseNode<
          _::FixVoid<T>, _::PromiseAndFulfillerAdapter<T>>>(*wrapper));
  auto promise = _::PromiseNode::to<_::ReducePromises<T>>(
      _::maybeChain(kj::mv(intermediate), implicitCast<T*>(nullptr), location));

  return PromiseFulfillerPair<T> { kj::mv(promise), kj::mv(wrapper) };
}

// =======================================================================================
// cross-thread stuff

namespace _ {  // (private)

class XThreadEvent: public PromiseNode,    // it's a PromiseNode in the requesting thread
                    private Event {        // it's an event in the target thread
public:
  XThreadEvent(ExceptionOrValue& result, const Executor& targetExecutor, EventLoop& loop,
               void* funcTracePtr, SourceLocation location);

  void tracePromise(TraceBuilder& builder, bool stopAtNextEvent) override;

protected:
  void ensureDoneOrCanceled();
  // MUST be called in destructor of subclasses to make sure the object is not destroyed while
  // still being accessed by the other thread. (This can't be placed in ~XThreadEvent() because
  // that destructor doesn't run until the subclass has already been destroyed.)

  virtual kj::Maybe<OwnPromiseNode> execute() = 0;
  // Run the function. If the function returns a promise, returns the inner PromiseNode, otherwise
  // returns null.

  // implements PromiseNode ----------------------------------------------------
  void onReady(Event* event) noexcept override;

private:
  ExceptionOrValue& result;
  void* funcTracePtr;

  kj::Own<const Executor> targetExecutor;
  Maybe<const Executor&> replyExecutor;  // If executeAsync() was used.

  kj::Maybe<OwnPromiseNode> promiseNode;
  // Accessed only in target thread.

  ListLink<XThreadEvent> targetLink;
  // Membership in one of the linked lists in the target Executor's work list or cancel list. These
  // fields are protected by the target Executor's mutex.

  enum {
    UNUSED,
    // Object was never queued on another thread.

    QUEUED,
    // Target thread has not yet dequeued the event from the state.start list. The requesting
    // thread can cancel execution by removing the event from the list.

    EXECUTING,
    // Target thread has dequeued the event from state.start and moved it to state.executing. To
    // cancel, the requesting thread must add the event to the state.cancel list and change the
    // state to CANCELING.

    CANCELING,
    // Requesting thread is trying to cancel this event. The target thread will change the state to
    // `DONE` once canceled.

    DONE
    // Target thread has completed handling this event and will not touch it again. The requesting
    // thread can safely delete the object. The `state` is updated to `DONE` using an atomic
    // release operation after ensuring that the event will not be touched again, so that the
    // requesting can safely skip locking if it observes the state is already DONE.
  } state = UNUSED;
  // State, which is also protected by `targetExecutor`'s mutex.

  ListLink<XThreadEvent> replyLink;
  // Membership in `replyExecutor`'s reply list. Protected by `replyExecutor`'s mutex. The
  // executing thread places the event in the reply list near the end of the `EXECUTING` state.
  // Because the thread cannot lock two mutexes at once, it's possible that the reply executor
  // will receive the reply while the event is still listed in the EXECUTING state, but it can
  // ignore the state and proceed with the result.

  OnReadyEvent onReadyEvent;
  // Accessed only in requesting thread.

  friend class kj::Executor;

  void done();
  // Sets the state to `DONE` and notifies the originating thread that this event is done. Do NOT
  // call under lock.

  void sendReply();
  // Notifies the originating thread that this event is done, but doesn't set the state to DONE
  // yet. Do NOT call under lock.

  void setDoneState();
  // Assigns `state` to `DONE`, being careful to use an atomic-release-store if needed. This must
  // only be called in the destination thread, and must either be called under lock, or the thread
  // must take the lock and release it again shortly after setting the state (because some threads
  // may be waiting on the DONE state using a conditional wait on the mutex). After calling
  // setDoneState(), the destination thread MUST NOT touch this object ever again; it now belongs
  // solely to the requesting thread.

  void setDisconnected();
  // Sets the result to a DISCONNECTED exception indicating that the target event loop exited.

  class DelayedDoneHack;

  // implements Event ----------------------------------------------------------
  Maybe<Own<Event>> fire() override;
  // If called with promiseNode == nullptr, it's time to call execute(). If promiseNode != nullptr,
  // then it just indicated readiness and we need to get its result.

  void traceEvent(TraceBuilder& builder) override;
};

template <typename Func, typename = _::FixVoid<_::ReturnType<Func, void>>>
class XThreadEventImpl final: public XThreadEvent {
  // Implementation for a function that does not return a Promise.
public:
  XThreadEventImpl(Func&& func, const Executor& target, EventLoop& loop, SourceLocation location)
      : XThreadEvent(result, target, loop, GetFunctorStartAddress<>::apply(func), location),
        func(kj::fwd<Func>(func)) {}
  ~XThreadEventImpl() noexcept(false) { ensureDoneOrCanceled(); }
  void destroy() override { freePromise(this); }

  typedef _::FixVoid<_::ReturnType<Func, void>> ResultT;

  kj::Maybe<_::OwnPromiseNode> execute() override {
    result.value = MaybeVoidCaller<Void, FixVoid<decltype(func())>>::apply(func, Void());
    return nullptr;
  }

  // implements PromiseNode ----------------------------------------------------
  void get(ExceptionOrValue& output) noexcept override {
    output.as<ResultT>() = kj::mv(result);
  }

private:
  Func func;
  ExceptionOr<ResultT> result;
  friend Executor;
};

template <typename Func, typename T>
class XThreadEventImpl<Func, Promise<T>> final: public XThreadEvent {
  // Implementation for a function that DOES return a Promise.
public:
  XThreadEventImpl(Func&& func, const Executor& target, EventLoop& loop, SourceLocation location)
      : XThreadEvent(result, target, loop, GetFunctorStartAddress<>::apply(func), location),
        func(kj::fwd<Func>(func)) {}
  ~XThreadEventImpl() noexcept(false) { ensureDoneOrCanceled(); }
  void destroy() override { freePromise(this); }

  typedef _::FixVoid<_::UnwrapPromise<PromiseForResult<Func, void>>> ResultT;

  kj::Maybe<_::OwnPromiseNode> execute() override {
    auto result = _::PromiseNode::from(func());
    KJ_IREQUIRE(result.get() != nullptr);
    return kj::mv(result);
  }

  // implements PromiseNode ----------------------------------------------------
  void get(ExceptionOrValue& output) noexcept override {
    output.as<ResultT>() = kj::mv(result);
  }

private:
  Func func;
  ExceptionOr<ResultT> result;
  friend Executor;
};

}  // namespace _ (private)

template <typename Func>
_::UnwrapPromise<PromiseForResult<Func, void>> Executor::executeSync(
    Func&& func, SourceLocation location) const {
  _::XThreadEventImpl<Func> event(kj::fwd<Func>(func), *this, getLoop(), location);
  send(event, true);
  return convertToReturn(kj::mv(event.result));
}

template <typename Func>
PromiseForResult<Func, void> Executor::executeAsync(Func&& func, SourceLocation location) const {
  // HACK: We call getLoop() here, rather than have XThreadEvent's constructor do it, so that if it
  //   throws we don't crash due to `allocPromise()` being `noexcept`.
  auto event = _::allocPromise<_::XThreadEventImpl<Func>>(
      kj::fwd<Func>(func), *this, getLoop(), location);
  send(*event, false);
  return _::PromiseNode::to<PromiseForResult<Func, void>>(kj::mv(event));
}

// -----------------------------------------------------------------------------

namespace _ {  // (private)

template <typename T>
class XThreadFulfiller;

class XThreadPaf: public PromiseNode {
public:
  XThreadPaf();
  virtual ~XThreadPaf() noexcept(false);
  void destroy() override;

  // implements PromiseNode ----------------------------------------------------
  void onReady(Event* event) noexcept override;
  void tracePromise(TraceBuilder& builder, bool stopAtNextEvent) override;

private:
  enum {
    WAITING,
    // Not yet fulfilled, and the waiter is still waiting.
    //
    // Starting from this state, the state may transition to either FULFILLING or CANCELED
    // using an atomic compare-and-swap.

    FULFILLING,
    // The fulfiller thread atomically transitions the state from WAITING to FULFILLING when it
    // wishes to fulfill the promise. By doing so, it guarantees that the `executor` will not
    // disappear out from under it. It then fills in the result value, locks the executor mutex,
    // adds the object to the executor's list of fulfilled XThreadPafs, changes the state to
    // FULFILLED, and finally unlocks the mutex.
    //
    // If the waiting thread tries to cancel but discovers the object in this state, then it
    // must perform a conditional wait on the executor mutex to await the state becoming FULFILLED.
    // It can then delete the object.

    FULFILLED,
    // The fulfilling thread has completed filling in the result value and inserting the object
    // into the waiting thread's executor event queue. Moreover, the fulfilling thread no longer
    // holds any pointers to this object. The waiting thread is responsible for deleting it.

    DISPATCHED,
    // The object reached FULFILLED state, and then was dispatched from the waiting thread's
    // executor's event queue. Therefore, the object is completely owned by the waiting thread with
    // no need to lock anything.

    CANCELED
    // The waiting thread atomically transitions the state from WAITING to CANCELED if it is no
    // longer listening. In this state, it is the fulfiller thread's responsibility to destroy the
    // object.
  } state;

  const Executor& executor;
  // Executor of the waiting thread. Only guaranteed to be valid when state is `WAITING` or
  // `FULFILLING`. After any other state has been reached, this reference may be invalidated.

  ListLink<XThreadPaf> link;
  // In the FULFILLING/FULFILLED states, the object is placed in a linked list within the waiting
  // thread's executor. In those states, these pointers are guarded by said executor's mutex.

  OnReadyEvent onReadyEvent;

  class FulfillScope;

  static kj::Exception unfulfilledException();
  // Construct appropriate exception to use to reject an unfulfilled XThreadPaf.

  template <typename T>
  friend class XThreadFulfiller;
  friend Executor;
};

template <typename T>
class XThreadPafImpl final: public XThreadPaf {
public:
  // implements PromiseNode ----------------------------------------------------
  void get(ExceptionOrValue& output) noexcept override {
    output.as<FixVoid<T>>() = kj::mv(result);
  }

private:
  ExceptionOr<FixVoid<T>> result;

  friend class XThreadFulfiller<T>;
};

class XThreadPaf::FulfillScope {
  // Create on stack while setting `XThreadPafImpl<T>::result`.
  //
  // This ensures that:
  // - Only one call is carried out, even if multiple threads try to fulfill concurrently.
  // - The waiting thread is correctly signaled.
public:
  FulfillScope(XThreadPaf** pointer);
  // Atomically nulls out *pointer and takes ownership of the pointer.

  ~FulfillScope() noexcept(false);

  KJ_DISALLOW_COPY_AND_MOVE(FulfillScope);

  bool shouldFulfill() { return obj != nullptr; }

  template <typename T>
  XThreadPafImpl<T>* getTarget() { return static_cast<XThreadPafImpl<T>*>(obj); }

private:
  XThreadPaf* obj;
};

template <typename T>
class XThreadFulfiller final: public CrossThreadPromiseFulfiller<T> {
public:
  XThreadFulfiller(XThreadPafImpl<T>* target): target(target) {}

  ~XThreadFulfiller() noexcept(false) {
    if (target != nullptr) {
      reject(XThreadPaf::unfulfilledException());
    }
  }
  void fulfill(FixVoid<T>&& value) const override {
    XThreadPaf::FulfillScope scope(&target);
    if (scope.shouldFulfill()) {
      scope.getTarget<T>()->result = kj::mv(value);
    }
  }
  void reject(Exception&& exception) const override {
    XThreadPaf::FulfillScope scope(&target);
    if (scope.shouldFulfill()) {
      scope.getTarget<T>()->result.addException(kj::mv(exception));
    }
  }
  bool isWaiting() const override {
    KJ_IF_MAYBE(t, target) {
#if _MSC_VER && !__clang__
      // Just assume 1-byte loads are atomic... on what kind of absurd platform would they not be?
      return t->state == XThreadPaf::WAITING;
#else
      return __atomic_load_n(&t->state, __ATOMIC_RELAXED) == XThreadPaf::WAITING;
#endif
    } else {
      return false;
    }
  }

private:
  mutable XThreadPaf* target;  // accessed using atomic ops
};

template <typename T>
class XThreadFulfiller<kj::Promise<T>> {
public:
  static_assert(sizeof(T) < 0,
      "newCrosssThreadPromiseAndFulfiller<Promise<T>>() is not currently supported");
  // TODO(someday): Is this worth supporting? Presumably, when someone calls `fulfill(somePromise)`,
  //   then `somePromise` should be assumed to be a promise owned by the fulfilling thread, not
  //   the waiting thread.
};

}  // namespace _ (private)

template <typename T>
PromiseCrossThreadFulfillerPair<T> newPromiseAndCrossThreadFulfiller() {
  kj::Own<_::XThreadPafImpl<T>, _::PromiseDisposer> node(new _::XThreadPafImpl<T>);
  auto fulfiller = kj::heap<_::XThreadFulfiller<T>>(node);
  return { _::PromiseNode::to<_::ReducePromises<T>>(kj::mv(node)), kj::mv(fulfiller) };
}

}  // namespace kj

#if KJ_HAS_COROUTINE

// =======================================================================================
// Coroutines TS integration with kj::Promise<T>.
//
// Here's a simple coroutine:
//
//   Promise<Own<AsyncIoStream>> connectToService(Network& n) {
//     auto a = co_await n.parseAddress(IP, PORT);
//     auto c = co_await a->connect();
//     co_return kj::mv(c);
//   }
//
// The presence of the co_await and co_return keywords tell the compiler it is a coroutine.
// Although it looks similar to a function, it has a couple large differences. First, everything
// that would normally live in the stack frame lives instead in a heap-based coroutine frame.
// Second, the coroutine has the ability to return from its scope without deallocating this frame
// (to suspend, in other words), and the ability to resume from its last suspension point.
//
// In order to know how to suspend, resume, and return from a coroutine, the compiler looks up a
// coroutine implementation type via a traits class parameterized by the coroutine return and
// parameter types. We'll name our coroutine implementation `kj::_::Coroutine<T>`,

namespace kj::_ { template <typename T> class Coroutine; }

// Specializing the appropriate traits class tells the compiler about `kj::_::Coroutine<T>`.

namespace KJ_COROUTINE_STD_NAMESPACE {

template <class T, class... Args>
struct coroutine_traits<kj::Promise<T>, Args...> {
  // `Args...` are the coroutine's parameter types.

  using promise_type = kj::_::Coroutine<T>;
  // The Coroutines TS calls this the "promise type". This makes sense when thinking of coroutines
  // returning `std::future<T>`, since the coroutine implementation would be a wrapper around
  // a `std::promise<T>`. It's extremely confusing from a KJ perspective, however, so I call it
  // the "coroutine implementation type" instead.
};

}  // namespace KJ_COROUTINE_STD_NAMESPACE

// Now when the compiler sees our `connectToService()` coroutine above, it default-constructs a
// `coroutine_traits<Promise<Own<AsyncIoStream>>, Network&>::promise_type`, or
// `kj::_::Coroutine<Own<AsyncIoStream>>`.
//
// The implementation object lives in the heap-allocated coroutine frame. It gets destroyed and
// deallocated when the frame does.

namespace kj::_ {

namespace stdcoro = KJ_COROUTINE_STD_NAMESPACE;

class CoroutineBase: public PromiseNode,
                     public Event {
public:
  CoroutineBase(stdcoro::coroutine_handle<> coroutine, ExceptionOrValue& resultRef,
                SourceLocation location);
  ~CoroutineBase() noexcept(false);
  KJ_DISALLOW_COPY_AND_MOVE(CoroutineBase);
  void destroy() override;

  auto initial_suspend() { return stdcoro::suspend_never(); }
  auto final_suspend() noexcept { return stdcoro::suspend_always(); }
  // These adjust the suspension behavior of coroutines immediately upon initiation, and immediately
  // after completion.
  //
  // The initial suspension point could allow us to defer the initial synchronous execution of a
  // coroutine -- everything before its first co_await, that is.
  //
  // The final suspension point is useful to delay deallocation of the coroutine frame to match the
  // lifetime of the enclosing promise.

  void unhandled_exception();

protected:
  class AwaiterBase;

  bool isWaiting() { return waiting; }
  void scheduleResumption() {
    onReadyEvent.arm();
    waiting = false;
  }

private:
  // -------------------------------------------------------
  // PromiseNode implementation

  void onReady(Event* event) noexcept override;
  void tracePromise(TraceBuilder& builder, bool stopAtNextEvent) override;

  // -------------------------------------------------------
  // Event implementation

  Maybe<Own<Event>> fire() override;
  void traceEvent(TraceBuilder& builder) override;

  stdcoro::coroutine_handle<> coroutine;
  ExceptionOrValue& resultRef;

  OnReadyEvent onReadyEvent;
  bool waiting = true;

  bool hasSuspendedAtLeastOnce = false;

  Maybe<PromiseNode&> promiseNodeForTrace;
  // Whenever this coroutine is suspended waiting on another promise, we keep a reference to that
  // promise so tracePromise()/traceEvent() can trace into it.

  UnwindDetector unwindDetector;

  struct DisposalResults {
    bool destructorRan = false;
    Maybe<Exception> exception;
  };
  Maybe<DisposalResults&> maybeDisposalResults;
  // Only non-null during destruction. Before calling coroutine.destroy(), our disposer sets this
  // to point to a DisposalResults on the stack so unhandled_exception() will have some place to
  // store unwind exceptions. We can't store them in this Coroutine, because we'll be destroyed once
  // coroutine.destroy() has returned. Our disposer then rethrows as needed.
};

template <typename Self, typename T>
class CoroutineMixin;
// CRTP mixin, covered later.

template <typename T>
class Coroutine final: public CoroutineBase,
                       public CoroutineMixin<Coroutine<T>, T> {
  // The standard calls this the `promise_type` object. We can call this the "coroutine
  // implementation object" since the word promise means different things in KJ and std styles. This
  // is where we implement how a `kj::Promise<T>` is returned from a coroutine, and how that promise
  // is later fulfilled. We also fill in a few lifetime-related details.
  //
  // The implementation object is also where we can customize memory allocation of coroutine frames,
  // by implementing a member `operator new(size_t, Args...)` (same `Args...` as in
  // coroutine_traits).
  //
  // We can also customize how await-expressions are transformed within `kj::Promise<T>`-based
  // coroutines by implementing an `await_transform(P)` member function, where `P` is some type for
  // which we want to implement co_await support, e.g. `kj::Promise<U>`. This feature allows us to
  // provide an optimized `kj::EventLoop` integration when the coroutine's return type and the
  // await-expression's type are both `kj::Promise` instantiations -- see further comments under
  // `await_transform()`.

public:
  using Handle = stdcoro::coroutine_handle<Coroutine<T>>;

  Coroutine(SourceLocation location = {})
      : CoroutineBase(Handle::from_promise(*this), result, location) {}

  Promise<T> get_return_object() {
    // Called after coroutine frame construction and before initial_suspend() to create the
    // coroutine's return object. `this` itself lives inside the coroutine frame, and we arrange for
    // the returned Promise<T> to own `this` via a custom Disposer and by always leaving the
    // coroutine in a suspended state.
    return PromiseNode::to<Promise<T>>(OwnPromiseNode(this));
  }

public:
  template <typename U>
  class Awaiter;

  template <typename U>
  Awaiter<U> await_transform(kj::Promise<U>& promise) { return Awaiter<U>(kj::mv(promise)); }
  template <typename U>
  Awaiter<U> await_transform(kj::Promise<U>&& promise) { return Awaiter<U>(kj::mv(promise)); }
  // Called when someone writes `co_await promise`, where `promise` is a kj::Promise<U>. We return
  // an Awaiter<U>, which implements coroutine suspension and resumption in terms of the KJ async
  // event system.
  //
  // There is another hook we could implement: an `operator co_await()` free function. However, a
  // free function would be unaware of the type of the enclosing coroutine. Since Awaiter<U> is a
  // member class template of Coroutine<T>, it is able to implement an
  // `await_suspend(Coroutine<T>::Handle)` override, providing it type-safe access to our enclosing
  // coroutine's PromiseNode. An `operator co_await()` free function would have to implement
  // a type-erased `await_suspend(stdcoro::coroutine_handle<void>)` override, and implement
  // suspension and resumption in terms of .then(). Yuck!

private:
  // -------------------------------------------------------
  // PromiseNode implementation

  void get(ExceptionOrValue& output) noexcept override {
    output.as<FixVoid<T>>() = kj::mv(result);
  }

  void fulfill(FixVoid<T>&& value) {
    // Called by the return_value()/return_void() functions in our mixin class.

    if (isWaiting()) {
      result = kj::mv(value);
      scheduleResumption();
    }
  }

  ExceptionOr<FixVoid<T>> result;

  friend class CoroutineMixin<Coroutine<T>, T>;
};

template <typename Self, typename T>
class CoroutineMixin {
public:
  void return_value(T value) {
    static_cast<Self*>(this)->fulfill(kj::mv(value));
  }
};
template <typename Self>
class CoroutineMixin<Self, void> {
public:
  void return_void() {
    static_cast<Self*>(this)->fulfill(_::Void());
  }
};
// The Coroutines spec has no `_::FixVoid<T>` equivalent to unify valueful and valueless co_return
// statements, and programs are ill-formed if the coroutine implementation object (Coroutine<T>) has
// both a `return_value()` and `return_void()`. No amount of EnableIffery can get around it, so
// these return_* functions live in a CRTP mixin.

class CoroutineBase::AwaiterBase {
public:
  explicit AwaiterBase(OwnPromiseNode node);
  AwaiterBase(AwaiterBase&&);
  ~AwaiterBase() noexcept(false);
  KJ_DISALLOW_COPY(AwaiterBase);

  bool await_ready() const { return false; }
  // This could return "`node->get()` is safe to call" instead, which would make suspension-less
  // co_awaits possible for immediately-fulfilled promises. However, we need an Event to figure that
  // out, and we won't have access to the Coroutine Event until await_suspend() is called. So, we
  // must return false here. Fortunately, await_suspend() has a trick up its sleeve to enable
  // suspension-less co_awaits.

protected:
  void getImpl(ExceptionOrValue& result, void* awaitedAt);
  bool awaitSuspendImpl(CoroutineBase& coroutineEvent);

private:
  UnwindDetector unwindDetector;
  OwnPromiseNode node;

  Maybe<CoroutineBase&> maybeCoroutineEvent;
  // If we do suspend waiting for our wrapped promise, we store a reference to `node` in our
  // enclosing Coroutine for tracing purposes. To guard against any edge cases where an async stack
  // trace is generated when an Awaiter was destroyed without Coroutine::fire() having been called,
  // we need our own reference to the enclosing Coroutine. (I struggle to think up any such
  // scenarios, but perhaps they could occur when destroying a suspended coroutine.)
};

template <typename T>
template <typename U>
class Coroutine<T>::Awaiter: public AwaiterBase {
  // Wrapper around a co_await'ed promise and some storage space for the result of that promise.
  // The compiler arranges to call our await_suspend() to suspend, which arranges to be woken up
  // when the awaited promise is settled. Once that happens, the enclosing coroutine's Event
  // implementation resumes the coroutine, which transitively calls await_resume() to unwrap the
  // awaited promise result.

public:
  explicit Awaiter(Promise<U> promise): AwaiterBase(PromiseNode::from(kj::mv(promise))) {}

  U await_resume() KJ_NOINLINE {
    // This is marked noinline in order to ensure __builtin_return_address() is accurate for stack
    // trace purposes. In my experimentation, this method was not inlined anyway even in opt
    // builds, but I want to make sure it doesn't suddenly start being inlined later causing stack
    // traces to break. (I also tried always-inline, but this did not appear to cause the compiler
    // to inline the method -- perhaps a limitation of coroutines?)
#if __GNUC__
    getImpl(result, __builtin_return_address(0));
#elif _MSC_VER
    getImpl(result, _ReturnAddress());
#else
    #error "please implement for your compiler"
#endif
    auto value = kj::_::readMaybe(result.value);
    KJ_IASSERT(value != nullptr, "Neither exception nor value present.");
    return U(kj::mv(*value));
  }

  bool await_suspend(Coroutine::Handle coroutine) {
    return awaitSuspendImpl(coroutine.promise());
  }

private:
  ExceptionOr<FixVoid<U>> result;
};

#undef KJ_COROUTINE_STD_NAMESPACE

}  // namespace kj::_ (private)

#endif  // KJ_HAS_COROUTINE

KJ_END_HEADER
