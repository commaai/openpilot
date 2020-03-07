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

#ifndef KJ_ASYNC_H_
#define KJ_ASYNC_H_

#if defined(__GNUC__) && !KJ_HEADER_WARNINGS
#pragma GCC system_header
#endif

#include "async-prelude.h"
#include "exception.h"
#include "refcount.h"

namespace kj {

class EventLoop;
class WaitScope;

template <typename T>
class Promise;
template <typename T>
class ForkedPromise;
template <typename T>
class PromiseFulfiller;
template <typename T>
struct PromiseFulfillerPair;

template <typename Func, typename T>
using PromiseForResult = Promise<_::JoinPromises<_::ReturnType<Func, T>>>;
// Evaluates to the type of Promise for the result of calling functor type Func with parameter type
// T.  If T is void, then the promise is for the result of calling Func with no arguments.  If
// Func itself returns a promise, the promises are joined, so you never get Promise<Promise<T>>.

// =======================================================================================
// Promises

template <typename T>
class Promise: protected _::PromiseBase {
  // The basic primitive of asynchronous computation in KJ.  Similar to "futures", but designed
  // specifically for event loop concurrency.  Similar to E promises and JavaScript Promises/A.
  //
  // A Promise represents a promise to produce a value of type T some time in the future.  Once
  // that value has been produced, the promise is "fulfilled".  Alternatively, a promise can be
  // "broken", with an Exception describing what went wrong.  You may implicitly convert a value of
  // type T to an already-fulfilled Promise<T>.  You may implicitly convert the constant
  // `kj::READY_NOW` to an already-fulfilled Promise<void>.  You may also implicitly convert a
  // `kj::Exception` to an already-broken promise of any type.
  //
  // Promises are linear types -- they are moveable but not copyable.  If a Promise is destroyed
  // or goes out of scope (without being moved elsewhere), any ongoing asynchronous operations
  // meant to fulfill the promise will be canceled if possible.  All methods of `Promise` (unless
  // otherwise noted) actually consume the promise in the sense of move semantics.  (Arguably they
  // should be rvalue-qualified, but at the time this interface was created compilers didn't widely
  // support that yet and anyway it would be pretty ugly typing kj::mv(promise).whatever().)  If
  // you want to use one Promise in two different places, you must fork it with `fork()`.
  //
  // To use the result of a Promise, you must call `then()` and supply a callback function to
  // call with the result.  `then()` returns another promise, for the result of the callback.
  // Any time that this would result in Promise<Promise<T>>, the promises are collapsed into a
  // simple Promise<T> that first waits for the outer promise, then the inner.  Example:
  //
  //     // Open a remote file, read the content, and then count the
  //     // number of lines of text.
  //     // Note that none of the calls here block.  `file`, `content`
  //     // and `lineCount` are all initialized immediately before any
  //     // asynchronous operations occur.  The lambda callbacks are
  //     // called later.
  //     Promise<Own<File>> file = openFtp("ftp://host/foo/bar");
  //     Promise<String> content = file.then(
  //         [](Own<File> file) -> Promise<String> {
  //           return file.readAll();
  //         });
  //     Promise<int> lineCount = content.then(
  //         [](String text) -> int {
  //           uint count = 0;
  //           for (char c: text) count += (c == '\n');
  //           return count;
  //         });
  //
  // For `then()` to work, the current thread must have an active `EventLoop`.  Each callback
  // is scheduled to execute in that loop.  Since `then()` schedules callbacks only on the current
  // thread's event loop, you do not need to worry about two callbacks running at the same time.
  // You will need to set up at least one `EventLoop` at the top level of your program before you
  // can use promises.
  //
  // To adapt a non-Promise-based asynchronous API to promises, use `newAdaptedPromise()`.
  //
  // Systems using promises should consider supporting the concept of "pipelining".  Pipelining
  // means allowing a caller to start issuing method calls against a promised object before the
  // promise has actually been fulfilled.  This is particularly useful if the promise is for a
  // remote object living across a network, as this can avoid round trips when chaining a series
  // of calls.  It is suggested that any class T which supports pipelining implement a subclass of
  // Promise<T> which adds "eventual send" methods -- methods which, when called, say "please
  // invoke the corresponding method on the promised value once it is available".  These methods
  // should in turn return promises for the eventual results of said invocations.  Cap'n Proto,
  // for example, implements the type `RemotePromise` which supports pipelining RPC requests -- see
  // `capnp/capability.h`.
  //
  // KJ Promises are based on E promises:
  //   http://wiki.erights.org/wiki/Walnut/Distributed_Computing#Promises
  //
  // KJ Promises are also inspired in part by the evolving standards for JavaScript/ECMAScript
  // promises, which are themselves influenced by E promises:
  //   http://promisesaplus.com/
  //   https://github.com/domenic/promises-unwrapping

public:
  Promise(_::FixVoid<T> value);
  // Construct an already-fulfilled Promise from a value of type T.  For non-void promises, the
  // parameter type is simply T.  So, e.g., in a function that returns `Promise<int>`, you can
  // say `return 123;` to return a promise that is already fulfilled to 123.
  //
  // For void promises, use `kj::READY_NOW` as the value, e.g. `return kj::READY_NOW`.

  Promise(kj::Exception&& e);
  // Construct an already-broken Promise.

  inline Promise(decltype(nullptr)) {}

  template <typename Func, typename ErrorFunc = _::PropagateException>
  PromiseForResult<Func, T> then(Func&& func, ErrorFunc&& errorHandler = _::PropagateException())
      KJ_WARN_UNUSED_RESULT;
  // Register a continuation function to be executed when the promise completes.  The continuation
  // (`func`) takes the promised value (an rvalue of type `T`) as its parameter.  The continuation
  // may return a new value; `then()` itself returns a promise for the continuation's eventual
  // result.  If the continuation itself returns a `Promise<U>`, then `then()` shall also return
  // a `Promise<U>` which first waits for the original promise, then executes the continuation,
  // then waits for the inner promise (i.e. it automatically "unwraps" the promise).
  //
  // In all cases, `then()` returns immediately.  The continuation is executed later.  The
  // continuation is always executed on the same EventLoop (and, therefore, the same thread) which
  // called `then()`, therefore no synchronization is necessary on state shared by the continuation
  // and the surrounding scope.  If no EventLoop is running on the current thread, `then()` throws
  // an exception.
  //
  // You may also specify an error handler continuation as the second parameter.  `errorHandler`
  // must be a functor taking a parameter of type `kj::Exception&&`.  It must return the same
  // type as `func` returns (except when `func` returns `Promise<U>`, in which case `errorHandler`
  // may return either `Promise<U>` or just `U`).  The default error handler simply propagates the
  // exception to the returned promise.
  //
  // Either `func` or `errorHandler` may, of course, throw an exception, in which case the promise
  // is broken.  When compiled with -fno-exceptions, the framework will still detect when a
  // recoverable exception was thrown inside of a continuation and will consider the promise
  // broken even though a (presumably garbage) result was returned.
  //
  // If the returned promise is destroyed before the callback runs, the callback will be canceled
  // (it will never run).
  //
  // Note that `then()` -- like all other Promise methods -- consumes the promise on which it is
  // called, in the sense of move semantics.  After returning, the original promise is no longer
  // valid, but `then()` returns a new promise.
  //
  // *Advanced implementation tips:*  Most users will never need to worry about the below, but
  // it is good to be aware of.
  //
  // As an optimization, if the callback function `func` does _not_ return another promise, then
  // execution of `func` itself may be delayed until its result is known to be needed.  The
  // expectation here is that `func` is just doing some transformation on the results, not
  // scheduling any other actions, therefore the system doesn't need to be proactive about
  // evaluating it.  This way, a chain of trivial then() transformations can be executed all at
  // once without repeatedly re-scheduling through the event loop.  Use the `eagerlyEvaluate()`
  // method to suppress this behavior.
  //
  // On the other hand, if `func` _does_ return another promise, then the system evaluates `func`
  // as soon as possible, because the promise it returns might be for a newly-scheduled
  // long-running asynchronous task.
  //
  // As another optimization, when a callback function registered with `then()` is actually
  // scheduled, it is scheduled to occur immediately, preempting other work in the event queue.
  // This allows a long chain of `then`s to execute all at once, improving cache locality by
  // clustering operations on the same data.  However, this implies that starvation can occur
  // if a chain of `then()`s takes a very long time to execute without ever stopping to wait for
  // actual I/O.  To solve this, use `kj::evalLater()` to yield control; this way, all other events
  // in the queue will get a chance to run before your callback is executed.

  Promise<void> ignoreResult() KJ_WARN_UNUSED_RESULT { return then([](T&&) {}); }
  // Convenience method to convert the promise to a void promise by ignoring the return value.
  //
  // You must still wait on the returned promise if you want the task to execute.

  template <typename ErrorFunc>
  Promise<T> catch_(ErrorFunc&& errorHandler) KJ_WARN_UNUSED_RESULT;
  // Equivalent to `.then(identityFunc, errorHandler)`, where `identifyFunc` is a function that
  // just returns its input.

  T wait(WaitScope& waitScope);
  // Run the event loop until the promise is fulfilled, then return its result.  If the promise
  // is rejected, throw an exception.
  //
  // wait() is primarily useful at the top level of a program -- typically, within the function
  // that allocated the EventLoop.  For example, a program that performs one or two RPCs and then
  // exits would likely use wait() in its main() function to wait on each RPC.  On the other hand,
  // server-side code generally cannot use wait(), because it has to be able to accept multiple
  // requests at once.
  //
  // If the promise is rejected, `wait()` throws an exception.  If the program was compiled without
  // exceptions (-fno-exceptions), this will usually abort.  In this case you really should first
  // use `then()` to set an appropriate handler for the exception case, so that the promise you
  // actually wait on never throws.
  //
  // `waitScope` is an object proving that the caller is in a scope where wait() is allowed.  By
  // convention, any function which might call wait(), or which might call another function which
  // might call wait(), must take `WaitScope&` as one of its parameters.  This is needed for two
  // reasons:
  // * `wait()` is not allowed during an event callback, because event callbacks are themselves
  //   called during some other `wait()`, and such recursive `wait()`s would only be able to
  //   complete in LIFO order, which might mean that the outer `wait()` ends up waiting longer
  //   than it is supposed to.  To prevent this, a `WaitScope` cannot be constructed or used during
  //   an event callback.
  // * Since `wait()` runs the event loop, unrelated event callbacks may execute before `wait()`
  //   returns.  This means that anyone calling `wait()` must be reentrant -- state may change
  //   around them in arbitrary ways.  Therefore, callers really need to know if a function they
  //   are calling might wait(), and the `WaitScope&` parameter makes this clear.
  //
  // TODO(someday):  Implement fibers, and let them call wait() even when they are handling an
  //   event.

  ForkedPromise<T> fork() KJ_WARN_UNUSED_RESULT;
  // Forks the promise, so that multiple different clients can independently wait on the result.
  // `T` must be copy-constructable for this to work.  Or, in the special case where `T` is
  // `Own<U>`, `U` must have a method `Own<U> addRef()` which returns a new reference to the same
  // (or an equivalent) object (probably implemented via reference counting).

  _::SplitTuplePromise<T> split();
  // Split a promise for a tuple into a tuple of promises.
  //
  // E.g. if you have `Promise<kj::Tuple<T, U>>`, `split()` returns
  // `kj::Tuple<Promise<T>, Promise<U>>`.

  Promise<T> exclusiveJoin(Promise<T>&& other) KJ_WARN_UNUSED_RESULT;
  // Return a new promise that resolves when either the original promise resolves or `other`
  // resolves (whichever comes first).  The promise that didn't resolve first is canceled.

  // TODO(someday): inclusiveJoin(), or perhaps just join(), which waits for both completions
  //   and produces a tuple?

  template <typename... Attachments>
  Promise<T> attach(Attachments&&... attachments) KJ_WARN_UNUSED_RESULT;
  // "Attaches" one or more movable objects (often, Own<T>s) to the promise, such that they will
  // be destroyed when the promise resolves.  This is useful when a promise's callback contains
  // pointers into some object and you want to make sure the object still exists when the callback
  // runs -- after calling then(), use attach() to add necessary objects to the result.

  template <typename ErrorFunc>
  Promise<T> eagerlyEvaluate(ErrorFunc&& errorHandler) KJ_WARN_UNUSED_RESULT;
  Promise<T> eagerlyEvaluate(decltype(nullptr)) KJ_WARN_UNUSED_RESULT;
  // Force eager evaluation of this promise.  Use this if you are going to hold on to the promise
  // for awhile without consuming the result, but you want to make sure that the system actually
  // processes it.
  //
  // `errorHandler` is a function that takes `kj::Exception&&`, like the second parameter to
  // `then()`, except that it must return void.  We make you specify this because otherwise it's
  // easy to forget to handle errors in a promise that you never use.  You may specify nullptr for
  // the error handler if you are sure that ignoring errors is fine, or if you know that you'll
  // eventually wait on the promise somewhere.

  template <typename ErrorFunc>
  void detach(ErrorFunc&& errorHandler);
  // Allows the promise to continue running in the background until it completes or the
  // `EventLoop` is destroyed.  Be careful when using this: since you can no longer cancel this
  // promise, you need to make sure that the promise owns all the objects it touches or make sure
  // those objects outlive the EventLoop.
  //
  // `errorHandler` is a function that takes `kj::Exception&&`, like the second parameter to
  // `then()`, except that it must return void.
  //
  // This function exists mainly to implement the Cap'n Proto requirement that RPC calls cannot be
  // canceled unless the callee explicitly permits it.

  kj::String trace();
  // Returns a dump of debug info about this promise.  Not for production use.  Requires RTTI.
  // This method does NOT consume the promise as other methods do.

private:
  Promise(bool, Own<_::PromiseNode>&& node): PromiseBase(kj::mv(node)) {}
  // Second parameter prevent ambiguity with immediate-value constructor.

  template <typename>
  friend class Promise;
  friend class EventLoop;
  template <typename U, typename Adapter, typename... Params>
  friend Promise<U> newAdaptedPromise(Params&&... adapterConstructorParams);
  template <typename U>
  friend PromiseFulfillerPair<U> newPromiseAndFulfiller();
  template <typename>
  friend class _::ForkHub;
  friend class _::TaskSetImpl;
  friend Promise<void> _::yield();
  friend class _::NeverDone;
  template <typename U>
  friend Promise<Array<U>> joinPromises(Array<Promise<U>>&& promises);
  friend Promise<void> joinPromises(Array<Promise<void>>&& promises);
};

template <typename T>
class ForkedPromise {
  // The result of `Promise::fork()` and `EventLoop::fork()`.  Allows branches to be created.
  // Like `Promise<T>`, this is a pass-by-move type.

public:
  inline ForkedPromise(decltype(nullptr)) {}

  Promise<T> addBranch();
  // Add a new branch to the fork.  The branch is equivalent to the original promise.

private:
  Own<_::ForkHub<_::FixVoid<T>>> hub;

  inline ForkedPromise(bool, Own<_::ForkHub<_::FixVoid<T>>>&& hub): hub(kj::mv(hub)) {}

  friend class Promise<T>;
  friend class EventLoop;
};

constexpr _::Void READY_NOW = _::Void();
// Use this when you need a Promise<void> that is already fulfilled -- this value can be implicitly
// cast to `Promise<void>`.

constexpr _::NeverDone NEVER_DONE = _::NeverDone();
// The opposite of `READY_NOW`, return this when the promise should never resolve.  This can be
// implicitly converted to any promise type.  You may also call `NEVER_DONE.wait()` to wait
// forever (useful for servers).

template <typename Func>
PromiseForResult<Func, void> evalLater(Func&& func) KJ_WARN_UNUSED_RESULT;
// Schedule for the given zero-parameter function to be executed in the event loop at some
// point in the near future.  Returns a Promise for its result -- or, if `func()` itself returns
// a promise, `evalLater()` returns a Promise for the result of resolving that promise.
//
// Example usage:
//     Promise<int> x = evalLater([]() { return 123; });
//
// The above is exactly equivalent to:
//     Promise<int> x = Promise<void>(READY_NOW).then([]() { return 123; });
//
// If the returned promise is destroyed before the callback runs, the callback will be canceled
// (never called).
//
// If you schedule several evaluations with `evalLater` during the same callback, they are
// guaranteed to be executed in order.

template <typename Func>
PromiseForResult<Func, void> evalNow(Func&& func) KJ_WARN_UNUSED_RESULT;
// Run `func()` and return a promise for its result. `func()` executes before `evalNow()` returns.
// If `func()` throws an exception, the exception is caught and wrapped in a promise -- this is the
// main reason why `evalNow()` is useful.

template <typename T>
Promise<Array<T>> joinPromises(Array<Promise<T>>&& promises);
// Join an array of promises into a promise for an array.

// =======================================================================================
// Hack for creating a lambda that holds an owned pointer.

template <typename Func, typename MovedParam>
class CaptureByMove {
public:
  inline CaptureByMove(Func&& func, MovedParam&& param)
      : func(kj::mv(func)), param(kj::mv(param)) {}

  template <typename... Params>
  inline auto operator()(Params&&... params)
      -> decltype(kj::instance<Func>()(kj::instance<MovedParam&&>(), kj::fwd<Params>(params)...)) {
    return func(kj::mv(param), kj::fwd<Params>(params)...);
  }

private:
  Func func;
  MovedParam param;
};

template <typename Func, typename MovedParam>
inline CaptureByMove<Func, Decay<MovedParam>> mvCapture(MovedParam&& param, Func&& func) {
  // Hack to create a "lambda" which captures a variable by moving it rather than copying or
  // referencing.  C++14 generalized captures should make this obsolete, but for now in C++11 this
  // is commonly needed for Promise continuations that own their state.  Example usage:
  //
  //    Own<Foo> ptr = makeFoo();
  //    Promise<int> promise = callRpc();
  //    promise.then(mvCapture(ptr, [](Own<Foo>&& ptr, int result) {
  //      return ptr->finish(result);
  //    }));

  return CaptureByMove<Func, Decay<MovedParam>>(kj::fwd<Func>(func), kj::mv(param));
}

// =======================================================================================
// Advanced promise construction

template <typename T>
class PromiseFulfiller {
  // A callback which can be used to fulfill a promise.  Only the first call to fulfill() or
  // reject() matters; subsequent calls are ignored.

public:
  virtual void fulfill(T&& value) = 0;
  // Fulfill the promise with the given value.

  virtual void reject(Exception&& exception) = 0;
  // Reject the promise with an error.

  virtual bool isWaiting() = 0;
  // Returns true if the promise is still unfulfilled and someone is potentially waiting for it.
  // Returns false if fulfill()/reject() has already been called *or* if the promise to be
  // fulfilled has been discarded and therefore the result will never be used anyway.

  template <typename Func>
  bool rejectIfThrows(Func&& func);
  // Call the function (with no arguments) and return true.  If an exception is thrown, call
  // `fulfiller.reject()` and then return false.  When compiled with exceptions disabled,
  // non-fatal exceptions are still detected and handled correctly.
};

template <>
class PromiseFulfiller<void> {
  // Specialization of PromiseFulfiller for void promises.  See PromiseFulfiller<T>.

public:
  virtual void fulfill(_::Void&& value = _::Void()) = 0;
  // Call with zero parameters.  The parameter is a dummy that only exists so that subclasses don't
  // have to specialize for <void>.

  virtual void reject(Exception&& exception) = 0;
  virtual bool isWaiting() = 0;

  template <typename Func>
  bool rejectIfThrows(Func&& func);
};

template <typename T, typename Adapter, typename... Params>
Promise<T> newAdaptedPromise(Params&&... adapterConstructorParams);
// Creates a new promise which owns an instance of `Adapter` which encapsulates the operation
// that will eventually fulfill the promise.  This is primarily useful for adapting non-KJ
// asynchronous APIs to use promises.
//
// An instance of `Adapter` will be allocated and owned by the returned `Promise`.  A
// `PromiseFulfiller<T>&` will be passed as the first parameter to the adapter's constructor,
// and `adapterConstructorParams` will be forwarded as the subsequent parameters.  The adapter
// is expected to perform some asynchronous operation and call the `PromiseFulfiller<T>` once
// it is finished.
//
// The adapter is destroyed when its owning Promise is destroyed.  This may occur before the
// Promise has been fulfilled.  In this case, the adapter's destructor should cancel the
// asynchronous operation.  Once the adapter is destroyed, the fulfillment callback cannot be
// called.
//
// An adapter implementation should be carefully written to ensure that it cannot accidentally
// be left unfulfilled permanently because of an exception.  Consider making liberal use of
// `PromiseFulfiller<T>::rejectIfThrows()`.

template <typename T>
struct PromiseFulfillerPair {
  Promise<_::JoinPromises<T>> promise;
  Own<PromiseFulfiller<T>> fulfiller;
};

template <typename T>
PromiseFulfillerPair<T> newPromiseAndFulfiller();
// Construct a Promise and a separate PromiseFulfiller which can be used to fulfill the promise.
// If the PromiseFulfiller is destroyed before either of its methods are called, the Promise is
// implicitly rejected.
//
// Although this function is easier to use than `newAdaptedPromise()`, it has the serious drawback
// that there is no way to handle cancellation (i.e. detect when the Promise is discarded).
//
// You can arrange to fulfill a promise with another promise by using a promise type for T.  E.g.
// `newPromiseAndFulfiller<Promise<U>>()` will produce a promise of type `Promise<U>` but the
// fulfiller will be of type `PromiseFulfiller<Promise<U>>`.  Thus you pass a `Promise<U>` to the
// `fulfill()` callback, and the promises are chained.

// =======================================================================================
// TaskSet

class TaskSet {
  // Holds a collection of Promise<void>s and ensures that each executes to completion.  Memory
  // associated with each promise is automatically freed when the promise completes.  Destroying
  // the TaskSet itself automatically cancels all unfinished promises.
  //
  // This is useful for "daemon" objects that perform background tasks which aren't intended to
  // fulfill any particular external promise, but which may need to be canceled (and thus can't
  // use `Promise::detach()`).  The daemon object holds a TaskSet to collect these tasks it is
  // working on.  This way, if the daemon itself is destroyed, the TaskSet is detroyed as well,
  // and everything the daemon is doing is canceled.

public:
  class ErrorHandler {
  public:
    virtual void taskFailed(kj::Exception&& exception) = 0;
  };

  TaskSet(ErrorHandler& errorHandler);
  // `loop` will be used to wait on promises.  `errorHandler` will be executed any time a task
  // throws an exception, and will execute within the given EventLoop.

  ~TaskSet() noexcept(false);

  void add(Promise<void>&& promise);

  kj::String trace();
  // Return debug info about all promises currently in the TaskSet.

private:
  Own<_::TaskSetImpl> impl;
};

// =======================================================================================
// The EventLoop class

class EventPort {
  // Interfaces between an `EventLoop` and events originating from outside of the loop's thread.
  // All such events come in through the `EventPort` implementation.
  //
  // An `EventPort` implementation may interface with low-level operating system APIs and/or other
  // threads.  You can also write an `EventPort` which wraps some other (non-KJ) event loop
  // framework, allowing the two to coexist in a single thread.

public:
  virtual bool wait() = 0;
  // Wait for an external event to arrive, sleeping if necessary.  Once at least one event has
  // arrived, queue it to the event loop (e.g. by fulfilling a promise) and return.
  //
  // This is called during `Promise::wait()` whenever the event queue becomes empty, in order to
  // wait for new events to populate the queue.
  //
  // It is safe to return even if nothing has actually been queued, so long as calling `wait()` in
  // a loop will eventually sleep.  (That is to say, false positives are fine.)
  //
  // Returns true if wake() has been called from another thread. (Precisely, returns true if
  // no previous call to wait `wait()` nor `poll()` has returned true since `wake()` was last
  // called.)

  virtual bool poll() = 0;
  // Check if any external events have arrived, but do not sleep.  If any events have arrived,
  // add them to the event queue (e.g. by fulfilling promises) before returning.
  //
  // This may be called during `Promise::wait()` when the EventLoop has been executing for a while
  // without a break but is still non-empty.
  //
  // Returns true if wake() has been called from another thread. (Precisely, returns true if
  // no previous call to wait `wait()` nor `poll()` has returned true since `wake()` was last
  // called.)

  virtual void setRunnable(bool runnable);
  // Called to notify the `EventPort` when the `EventLoop` has work to do; specifically when it
  // transitions from empty -> runnable or runnable -> empty.  This is typically useful when
  // integrating with an external event loop; if the loop is currently runnable then you should
  // arrange to call run() on it soon.  The default implementation does nothing.

  virtual void wake() const;
  // Wake up the EventPort's thread from another thread.
  //
  // Unlike all other methods on this interface, `wake()` may be called from another thread, hence
  // it is `const`.
  //
  // Technically speaking, `wake()` causes the target thread to cease sleeping and not to sleep
  // again until `wait()` or `poll()` has returned true at least once.
  //
  // The default implementation throws an UNIMPLEMENTED exception.
};

class EventLoop {
  // Represents a queue of events being executed in a loop.  Most code won't interact with
  // EventLoop directly, but instead use `Promise`s to interact with it indirectly.  See the
  // documentation for `Promise`.
  //
  // Each thread can have at most one current EventLoop.  To make an `EventLoop` current for
  // the thread, create a `WaitScope`.  Async APIs require that the thread has a current EventLoop,
  // or they will throw exceptions.  APIs that use `Promise::wait()` additionally must explicitly
  // be passed a reference to the `WaitScope` to make the caller aware that they might block.
  //
  // Generally, you will want to construct an `EventLoop` at the top level of your program, e.g.
  // in the main() function, or in the start function of a thread.  You can then use it to
  // construct some promises and wait on the result.  Example:
  //
  //     int main() {
  //       // `loop` becomes the official EventLoop for the thread.
  //       MyEventPort eventPort;
  //       EventLoop loop(eventPort);
  //
  //       // Now we can call an async function.
  //       Promise<String> textPromise = getHttp("http://example.com");
  //
  //       // And we can wait for the promise to complete.  Note that you can only use `wait()`
  //       // from the top level, not from inside a promise callback.
  //       String text = textPromise.wait();
  //       print(text);
  //       return 0;
  //     }
  //
  // Most applications that do I/O will prefer to use `setupAsyncIo()` from `async-io.h` rather
  // than allocate an `EventLoop` directly.

public:
  EventLoop();
  // Construct an `EventLoop` which does not receive external events at all.

  explicit EventLoop(EventPort& port);
  // Construct an `EventLoop` which receives external events through the given `EventPort`.

  ~EventLoop() noexcept(false);

  void run(uint maxTurnCount = maxValue);
  // Run the event loop for `maxTurnCount` turns or until there is nothing left to be done,
  // whichever comes first.  This never calls the `EventPort`'s `sleep()` or `poll()`.  It will
  // call the `EventPort`'s `setRunnable(false)` if the queue becomes empty.

  bool isRunnable();
  // Returns true if run() would currently do anything, or false if the queue is empty.

private:
  EventPort& port;

  bool running = false;
  // True while looping -- wait() is then not allowed.

  bool lastRunnableState = false;
  // What did we last pass to port.setRunnable()?

  _::Event* head = nullptr;
  _::Event** tail = &head;
  _::Event** depthFirstInsertPoint = &head;

  Own<_::TaskSetImpl> daemons;

  bool turn();
  void setRunnable(bool runnable);
  void enterScope();
  void leaveScope();

  friend void _::detach(kj::Promise<void>&& promise);
  friend void _::waitImpl(Own<_::PromiseNode>&& node, _::ExceptionOrValue& result,
                          WaitScope& waitScope);
  friend class _::Event;
  friend class WaitScope;
};

class WaitScope {
  // Represents a scope in which asynchronous programming can occur.  A `WaitScope` should usually
  // be allocated on the stack and serves two purposes:
  // * While the `WaitScope` exists, its `EventLoop` is registered as the current loop for the
  //   thread.  Most operations dealing with `Promise` (including all of its methods) do not work
  //   unless the thread has a current `EventLoop`.
  // * `WaitScope` may be passed to `Promise::wait()` to synchronously wait for a particular
  //   promise to complete.  See `Promise::wait()` for an extended discussion.

public:
  inline explicit WaitScope(EventLoop& loop): loop(loop) { loop.enterScope(); }
  inline ~WaitScope() { loop.leaveScope(); }
  KJ_DISALLOW_COPY(WaitScope);

private:
  EventLoop& loop;
  friend class EventLoop;
  friend void _::waitImpl(Own<_::PromiseNode>&& node, _::ExceptionOrValue& result,
                          WaitScope& waitScope);
};

}  // namespace kj

#include "async-inl.h"

#endif  // KJ_ASYNC_H_
