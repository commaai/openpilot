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

#ifndef KJ_ASYNC_H_
#error "Do not include this directly; include kj/async.h."
#include "async.h"  // help IDE parse this file
#endif

#ifndef KJ_ASYNC_INL_H_
#define KJ_ASYNC_INL_H_

#if defined(__GNUC__) && !KJ_HEADER_WARNINGS
#pragma GCC system_header
#endif

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

class Event {
  // An event waiting to be executed.  Not for direct use by applications -- promises use this
  // internally.

public:
  Event();
  ~Event() noexcept(false);
  KJ_DISALLOW_COPY(Event);

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

  kj::String trace();
  // Dump debug info about this event.

  virtual _::PromiseNode* getInnerForTrace();
  // If this event wraps a PromiseNode, get that node.  Used for debug tracing.
  // Default implementation returns nullptr.

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
};

class PromiseNode {
  // A Promise<T> contains a chain of PromiseNodes tracking the pending transformations.
  //
  // To reduce generated code bloat, PromiseNode is not a template.  Instead, it makes very hacky
  // use of pointers to ExceptionOrValue which actually point to ExceptionOr<T>, but are only
  // so down-cast in the few places that really need to be templated.  Luckily this is all
  // internal implementation details.

public:
  virtual void onReady(Event& event) noexcept = 0;
  // Arms the given event when ready.

  virtual void setSelfPointer(Own<PromiseNode>* selfPtr) noexcept;
  // Tells the node that `selfPtr` is the pointer that owns this node, and will continue to own
  // this node until it is destroyed or setSelfPointer() is called again.  ChainPromiseNode uses
  // this to shorten redundant chains.  The default implementation does nothing; only
  // ChainPromiseNode should implement this.

  virtual void get(ExceptionOrValue& output) noexcept = 0;
  // Get the result.  `output` points to an ExceptionOr<T> into which the result will be written.
  // Can only be called once, and only after the node is ready.  Must be called directly from the
  // event loop, with no application code on the stack.

  virtual PromiseNode* getInnerForTrace();
  // If this node wraps some other PromiseNode, get the wrapped node.  Used for debug tracing.
  // Default implementation returns nullptr.

protected:
  class OnReadyEvent {
    // Helper class for implementing onReady().

  public:
    void init(Event& newEvent);
    // Returns true if arm() was already called.

    void arm();
    // Arms the event if init() has already been called and makes future calls to init() return
    // true.

  private:
    Event* event = nullptr;
  };
};

// -------------------------------------------------------------------

class ImmediatePromiseNodeBase: public PromiseNode {
public:
  ImmediatePromiseNodeBase();
  ~ImmediatePromiseNodeBase() noexcept(false);

  void onReady(Event& event) noexcept override;
};

template <typename T>
class ImmediatePromiseNode final: public ImmediatePromiseNodeBase {
  // A promise that has already been resolved to an immediate value or exception.

public:
  ImmediatePromiseNode(ExceptionOr<T>&& result): result(kj::mv(result)) {}

  void get(ExceptionOrValue& output) noexcept override {
    output.as<T>() = kj::mv(result);
  }

private:
  ExceptionOr<T> result;
};

class ImmediateBrokenPromiseNode final: public ImmediatePromiseNodeBase {
public:
  ImmediateBrokenPromiseNode(Exception&& exception);

  void get(ExceptionOrValue& output) noexcept override;

private:
  Exception exception;
};

// -------------------------------------------------------------------

class AttachmentPromiseNodeBase: public PromiseNode {
public:
  AttachmentPromiseNodeBase(Own<PromiseNode>&& dependency);

  void onReady(Event& event) noexcept override;
  void get(ExceptionOrValue& output) noexcept override;
  PromiseNode* getInnerForTrace() override;

private:
  Own<PromiseNode> dependency;

  void dropDependency();

  template <typename>
  friend class AttachmentPromiseNode;
};

template <typename Attachment>
class AttachmentPromiseNode final: public AttachmentPromiseNodeBase {
  // A PromiseNode that holds on to some object (usually, an Own<T>, but could be any movable
  // object) until the promise resolves.

public:
  AttachmentPromiseNode(Own<PromiseNode>&& dependency, Attachment&& attachment)
      : AttachmentPromiseNodeBase(kj::mv(dependency)),
        attachment(kj::mv<Attachment>(attachment)) {}

  ~AttachmentPromiseNode() noexcept(false) {
    // We need to make sure the dependency is deleted before we delete the attachment because the
    // dependency may be using the attachment.
    dropDependency();
  }

private:
  Attachment attachment;
};

// -------------------------------------------------------------------

class PtmfHelper {
  // This class is a private helper for GetFunctorStartAddress. The class represents the internal
  // representation of a pointer-to-member-function.

  template <typename... ParamTypes>
  friend struct GetFunctorStartAddress;

#if __GNUG__

  void* ptr;
  ptrdiff_t adj;
  // Layout of a pointer-to-member-function used by GCC and compatible compilers.

  void* apply(void* obj) {
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

  void* apply(void* obj) { return nullptr; }
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
  TransformPromiseNodeBase(Own<PromiseNode>&& dependency, void* continuationTracePtr);

  void onReady(Event& event) noexcept override;
  void get(ExceptionOrValue& output) noexcept override;
  PromiseNode* getInnerForTrace() override;

private:
  Own<PromiseNode> dependency;
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
  TransformPromiseNode(Own<PromiseNode>&& dependency, Func&& func, ErrorFunc&& errorHandler)
      : TransformPromiseNodeBase(kj::mv(dependency),
            GetFunctorStartAddress<DepT&&>::apply(func)),
        func(kj::fwd<Func>(func)), errorHandler(kj::fwd<ErrorFunc>(errorHandler)) {}

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

class ForkBranchBase: public PromiseNode {
public:
  ForkBranchBase(Own<ForkHubBase>&& hub);
  ~ForkBranchBase() noexcept(false);

  void hubReady() noexcept;
  // Called by the hub to indicate that it is ready.

  // implements PromiseNode ------------------------------------------
  void onReady(Event& event) noexcept override;
  PromiseNode* getInnerForTrace() override;

protected:
  inline ExceptionOrValue& getHubResultRef();

  void releaseHub(ExceptionOrValue& output);
  // Release the hub.  If an exception is thrown, add it to `output`.

private:
  OnReadyEvent onReadyEvent;

  Own<ForkHubBase> hub;
  ForkBranchBase* next = nullptr;
  ForkBranchBase** prevPtr = nullptr;

  friend class ForkHubBase;
};

template <typename T> T copyOrAddRef(T& t) { return t; }
template <typename T> Own<T> copyOrAddRef(Own<T>& t) { return t->addRef(); }

template <typename T>
class ForkBranch final: public ForkBranchBase {
  // A PromiseNode that implements one branch of a fork -- i.e. one of the branches that receives
  // a const reference.

public:
  ForkBranch(Own<ForkHubBase>&& hub): ForkBranchBase(kj::mv(hub)) {}

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
  SplitBranch(Own<ForkHubBase>&& hub): ForkBranchBase(kj::mv(hub)) {}

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

class ForkHubBase: public Refcounted, protected Event {
public:
  ForkHubBase(Own<PromiseNode>&& inner, ExceptionOrValue& resultRef);

  inline ExceptionOrValue& getResultRef() { return resultRef; }

private:
  Own<PromiseNode> inner;
  ExceptionOrValue& resultRef;

  ForkBranchBase* headBranch = nullptr;
  ForkBranchBase** tailBranch = &headBranch;
  // Tail becomes null once the inner promise is ready and all branches have been notified.

  Maybe<Own<Event>> fire() override;
  _::PromiseNode* getInnerForTrace() override;

  friend class ForkBranchBase;
};

template <typename T>
class ForkHub final: public ForkHubBase {
  // A PromiseNode that implements the hub of a fork.  The first call to Promise::fork() replaces
  // the promise's outer node with a ForkHub, and subsequent calls add branches to that hub (if
  // possible).

public:
  ForkHub(Own<PromiseNode>&& inner): ForkHubBase(kj::mv(inner), result) {}

  Promise<_::UnfixVoid<T>> addBranch() {
    return Promise<_::UnfixVoid<T>>(false, kj::heap<ForkBranch<T>>(addRef(*this)));
  }

  _::SplitTuplePromise<T> split() {
    return splitImpl(MakeIndexes<tupleSize<T>()>());
  }

private:
  ExceptionOr<T> result;

  template <size_t... indexes>
  _::SplitTuplePromise<T> splitImpl(Indexes<indexes...>) {
    return kj::tuple(addSplit<indexes>()...);
  }

  template <size_t index>
  Promise<JoinPromises<typename SplitBranch<T, index>::Element>> addSplit() {
    return Promise<JoinPromises<typename SplitBranch<T, index>::Element>>(
        false, maybeChain(kj::heap<SplitBranch<T, index>>(addRef(*this)),
                          implicitCast<typename SplitBranch<T, index>::Element*>(nullptr)));
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
  explicit ChainPromiseNode(Own<PromiseNode> inner);
  ~ChainPromiseNode() noexcept(false);

  void onReady(Event& event) noexcept override;
  void setSelfPointer(Own<PromiseNode>* selfPtr) noexcept override;
  void get(ExceptionOrValue& output) noexcept override;
  PromiseNode* getInnerForTrace() override;

private:
  enum State {
    STEP1,
    STEP2
  };

  State state;

  Own<PromiseNode> inner;
  // In STEP1, a PromiseNode for a Promise<T>.
  // In STEP2, a PromiseNode for a T.

  Event* onReadyEvent = nullptr;
  Own<PromiseNode>* selfPtr = nullptr;

  Maybe<Own<Event>> fire() override;
};

template <typename T>
Own<PromiseNode> maybeChain(Own<PromiseNode>&& node, Promise<T>*) {
  return heap<ChainPromiseNode>(kj::mv(node));
}

template <typename T>
Own<PromiseNode>&& maybeChain(Own<PromiseNode>&& node, T*) {
  return kj::mv(node);
}

// -------------------------------------------------------------------

class ExclusiveJoinPromiseNode final: public PromiseNode {
public:
  ExclusiveJoinPromiseNode(Own<PromiseNode> left, Own<PromiseNode> right);
  ~ExclusiveJoinPromiseNode() noexcept(false);

  void onReady(Event& event) noexcept override;
  void get(ExceptionOrValue& output) noexcept override;
  PromiseNode* getInnerForTrace() override;

private:
  class Branch: public Event {
  public:
    Branch(ExclusiveJoinPromiseNode& joinNode, Own<PromiseNode> dependency);
    ~Branch() noexcept(false);

    bool get(ExceptionOrValue& output);
    // Returns true if this is the side that finished.

    Maybe<Own<Event>> fire() override;
    _::PromiseNode* getInnerForTrace() override;

  private:
    ExclusiveJoinPromiseNode& joinNode;
    Own<PromiseNode> dependency;
  };

  Branch left;
  Branch right;
  OnReadyEvent onReadyEvent;
};

// -------------------------------------------------------------------

class ArrayJoinPromiseNodeBase: public PromiseNode {
public:
  ArrayJoinPromiseNodeBase(Array<Own<PromiseNode>> promises,
                           ExceptionOrValue* resultParts, size_t partSize);
  ~ArrayJoinPromiseNodeBase() noexcept(false);

  void onReady(Event& event) noexcept override final;
  void get(ExceptionOrValue& output) noexcept override final;
  PromiseNode* getInnerForTrace() override final;

protected:
  virtual void getNoError(ExceptionOrValue& output) noexcept = 0;
  // Called to compile the result only in the case where there were no errors.

private:
  uint countLeft;
  OnReadyEvent onReadyEvent;

  class Branch final: public Event {
  public:
    Branch(ArrayJoinPromiseNodeBase& joinNode, Own<PromiseNode> dependency,
           ExceptionOrValue& output);
    ~Branch() noexcept(false);

    Maybe<Own<Event>> fire() override;
    _::PromiseNode* getInnerForTrace() override;

    Maybe<Exception> getPart();
    // Calls dependency->get(output).  If there was an exception, return it.

  private:
    ArrayJoinPromiseNodeBase& joinNode;
    Own<PromiseNode> dependency;
    ExceptionOrValue& output;
  };

  Array<Branch> branches;
};

template <typename T>
class ArrayJoinPromiseNode final: public ArrayJoinPromiseNodeBase {
public:
  ArrayJoinPromiseNode(Array<Own<PromiseNode>> promises,
                       Array<ExceptionOr<T>> resultParts)
      : ArrayJoinPromiseNodeBase(kj::mv(promises), resultParts.begin(), sizeof(ExceptionOr<T>)),
        resultParts(kj::mv(resultParts)) {}

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
  ArrayJoinPromiseNode(Array<Own<PromiseNode>> promises,
                       Array<ExceptionOr<_::Void>> resultParts);
  ~ArrayJoinPromiseNode();

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
  EagerPromiseNodeBase(Own<PromiseNode>&& dependency, ExceptionOrValue& resultRef);

  void onReady(Event& event) noexcept override;
  PromiseNode* getInnerForTrace() override;

private:
  Own<PromiseNode> dependency;
  OnReadyEvent onReadyEvent;

  ExceptionOrValue& resultRef;

  Maybe<Own<Event>> fire() override;
};

template <typename T>
class EagerPromiseNode final: public EagerPromiseNodeBase {
public:
  EagerPromiseNode(Own<PromiseNode>&& dependency)
      : EagerPromiseNodeBase(kj::mv(dependency), result) {}

  void get(ExceptionOrValue& output) noexcept override {
    output.as<T>() = kj::mv(result);
  }

private:
  ExceptionOr<T> result;
};

template <typename T>
Own<PromiseNode> spark(Own<PromiseNode>&& node) {
  // Forces evaluation of the given node to begin as soon as possible, even if no one is waiting
  // on it.
  return heap<EagerPromiseNode<T>>(kj::mv(node));
}

// -------------------------------------------------------------------

class AdapterPromiseNodeBase: public PromiseNode {
public:
  void onReady(Event& event) noexcept override;

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

}  // namespace _ (private)

// =======================================================================================

template <typename T>
Promise<T>::Promise(_::FixVoid<T> value)
    : PromiseBase(heap<_::ImmediatePromiseNode<_::FixVoid<T>>>(kj::mv(value))) {}

template <typename T>
Promise<T>::Promise(kj::Exception&& exception)
    : PromiseBase(heap<_::ImmediateBrokenPromiseNode>(kj::mv(exception))) {}

template <typename T>
template <typename Func, typename ErrorFunc>
PromiseForResult<Func, T> Promise<T>::then(Func&& func, ErrorFunc&& errorHandler) {
  typedef _::FixVoid<_::ReturnType<Func, T>> ResultT;

  Own<_::PromiseNode> intermediate =
      heap<_::TransformPromiseNode<ResultT, _::FixVoid<T>, Func, ErrorFunc>>(
          kj::mv(node), kj::fwd<Func>(func), kj::fwd<ErrorFunc>(errorHandler));
  return PromiseForResult<Func, T>(false,
      _::maybeChain(kj::mv(intermediate), implicitCast<ResultT*>(nullptr)));
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
Promise<T> Promise<T>::catch_(ErrorFunc&& errorHandler) {
  // then()'s ErrorFunc can only return a Promise if Func also returns a Promise. In this case,
  // Func is being filled in automatically. We want to make sure ErrorFunc can return a Promise,
  // but we don't want the extra overhead of promise chaining if ErrorFunc doesn't actually
  // return a promise. So we make our Func return match ErrorFunc.
  return then(_::IdentityFunc<decltype(errorHandler(instance<Exception&&>()))>(),
              kj::fwd<ErrorFunc>(errorHandler));
}

template <typename T>
T Promise<T>::wait(WaitScope& waitScope) {
  _::ExceptionOr<_::FixVoid<T>> result;

  waitImpl(kj::mv(node), result, waitScope);

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

template <>
inline void Promise<void>::wait(WaitScope& waitScope) {
  // Override <void> case to use throwRecoverableException().

  _::ExceptionOr<_::Void> result;

  waitImpl(kj::mv(node), result, waitScope);

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

template <typename T>
ForkedPromise<T> Promise<T>::fork() {
  return ForkedPromise<T>(false, refcounted<_::ForkHub<_::FixVoid<T>>>(kj::mv(node)));
}

template <typename T>
Promise<T> ForkedPromise<T>::addBranch() {
  return hub->addBranch();
}

template <typename T>
_::SplitTuplePromise<T> Promise<T>::split() {
  return refcounted<_::ForkHub<_::FixVoid<T>>>(kj::mv(node))->split();
}

template <typename T>
Promise<T> Promise<T>::exclusiveJoin(Promise<T>&& other) {
  return Promise(false, heap<_::ExclusiveJoinPromiseNode>(kj::mv(node), kj::mv(other.node)));
}

template <typename T>
template <typename... Attachments>
Promise<T> Promise<T>::attach(Attachments&&... attachments) {
  return Promise(false, kj::heap<_::AttachmentPromiseNode<Tuple<Attachments...>>>(
      kj::mv(node), kj::tuple(kj::fwd<Attachments>(attachments)...)));
}

template <typename T>
template <typename ErrorFunc>
Promise<T> Promise<T>::eagerlyEvaluate(ErrorFunc&& errorHandler) {
  // See catch_() for commentary.
  return Promise(false, _::spark<_::FixVoid<T>>(then(
      _::IdentityFunc<decltype(errorHandler(instance<Exception&&>()))>(),
      kj::fwd<ErrorFunc>(errorHandler)).node));
}

template <typename T>
Promise<T> Promise<T>::eagerlyEvaluate(decltype(nullptr)) {
  return Promise(false, _::spark<_::FixVoid<T>>(kj::mv(node)));
}

template <typename T>
kj::String Promise<T>::trace() {
  return PromiseBase::trace();
}

template <typename Func>
inline PromiseForResult<Func, void> evalLater(Func&& func) {
  return _::yield().then(kj::fwd<Func>(func), _::PropagateException());
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
Promise<Array<T>> joinPromises(Array<Promise<T>>&& promises) {
  return Promise<Array<T>>(false, kj::heap<_::ArrayJoinPromiseNode<T>>(
      KJ_MAP(p, promises) { return kj::mv(p.node); },
      heapArray<_::ExceptionOr<T>>(promises.size())));
}

// =======================================================================================

namespace _ {  // private

template <typename T>
class WeakFulfiller final: public PromiseFulfiller<T>, private kj::Disposer {
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
  KJ_DISALLOW_COPY(WeakFulfiller);

  static kj::Own<WeakFulfiller> make() {
    WeakFulfiller* ptr = new WeakFulfiller;
    return Own<WeakFulfiller>(ptr, *ptr);
  }

  void fulfill(FixVoid<T>&& value) override {
    if (inner != nullptr) {
      inner->fulfill(kj::mv(value));
    }
  }

  void reject(Exception&& exception) override {
    if (inner != nullptr) {
      inner->reject(kj::mv(exception));
    }
  }

  bool isWaiting() override {
    return inner != nullptr && inner->isWaiting();
  }

  void attach(PromiseFulfiller<T>& newInner) {
    inner = &newInner;
  }

  void detach(PromiseFulfiller<T>& from) {
    if (inner == nullptr) {
      // Already disposed.
      delete this;
    } else {
      KJ_IREQUIRE(inner == &from);
      inner = nullptr;
    }
  }

private:
  mutable PromiseFulfiller<T>* inner;

  WeakFulfiller(): inner(nullptr) {}

  void disposeImpl(void* pointer) const override {
    // TODO(perf): Factor some of this out so it isn't regenerated for every fulfiller type?

    if (inner == nullptr) {
      // Already detached.
      delete this;
    } else {
      if (inner->isWaiting()) {
        inner->reject(kj::Exception(kj::Exception::Type::FAILED, __FILE__, __LINE__,
            kj::heapString("PromiseFulfiller was destroyed without fulfilling the promise.")));
      }
      inner = nullptr;
    }
  }
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
Promise<T> newAdaptedPromise(Params&&... adapterConstructorParams) {
  return Promise<T>(false, heap<_::AdapterPromiseNode<_::FixVoid<T>, Adapter>>(
      kj::fwd<Params>(adapterConstructorParams)...));
}

template <typename T>
PromiseFulfillerPair<T> newPromiseAndFulfiller() {
  auto wrapper = _::WeakFulfiller<T>::make();

  Own<_::PromiseNode> intermediate(
      heap<_::AdapterPromiseNode<_::FixVoid<T>, _::PromiseAndFulfillerAdapter<T>>>(*wrapper));
  Promise<_::JoinPromises<T>> promise(false,
      _::maybeChain(kj::mv(intermediate), implicitCast<T*>(nullptr)));

  return PromiseFulfillerPair<T> { kj::mv(promise), kj::mv(wrapper) };
}

}  // namespace kj

#endif  // KJ_ASYNC_INL_H_
