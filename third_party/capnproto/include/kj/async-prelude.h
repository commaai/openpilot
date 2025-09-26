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

// This file contains a bunch of internal declarations that must appear before async.h can start.
// We don't define these directly in async.h because it makes the file hard to read.

#pragma once

#include <kj/exception.h>
#include <kj/tuple.h>
#include <kj/source-location.h>

// Detect whether or not we should enable kj::Promise<T> coroutine integration.
//
// TODO(someday): Support coroutines with -fno-exceptions.
#if !KJ_NO_EXCEPTIONS
#ifdef __has_include
#if (__cpp_impl_coroutine >= 201902L) && __has_include(<coroutine>)
// C++20 Coroutines detected.
#include <coroutine>
#define KJ_HAS_COROUTINE 1
#define KJ_COROUTINE_STD_NAMESPACE std
#elif (__cpp_coroutines >= 201703L) && __has_include(<experimental/coroutine>)
// Coroutines TS detected.
#include <experimental/coroutine>
#define KJ_HAS_COROUTINE 1
#define KJ_COROUTINE_STD_NAMESPACE std::experimental
#endif
#endif
#endif

KJ_BEGIN_HEADER

namespace kj {

class EventLoop;
template <typename T>
class Promise;
class WaitScope;
class TaskSet;

Promise<void> joinPromises(Array<Promise<void>>&& promises, SourceLocation location = {});
Promise<void> joinPromisesFailFast(Array<Promise<void>>&& promises, SourceLocation location = {});
// Out-of-line <void> specialization of template function defined in async.h.

namespace _ {  // private

template <typename T>
Promise<T> chainPromiseType(T*);
template <typename T>
Promise<T> chainPromiseType(Promise<T>*);

template <typename T>
using ChainPromises = decltype(chainPromiseType((T*)nullptr));
// Constructs a promise for T, reducing double-promises. That is, if T is Promise<U>, resolves to
// Promise<U>, otherwise resolves to Promise<T>.

template <typename T>
Promise<T> reducePromiseType(T*, ...);
template <typename T>
Promise<T> reducePromiseType(Promise<T>*, ...);
template <typename T, typename Reduced = decltype(T::reducePromise(kj::instance<Promise<T>>()))>
Reduced reducePromiseType(T*, bool);

template <typename T>
using ReducePromises = decltype(reducePromiseType((T*)nullptr, false));
// Like ChainPromises, but also takes into account whether T has a method `reducePromise` that
// reduces Promise<T> to something else. In particular this allows Promise<capnp::RemotePromise<U>>
// to reduce to capnp::RemotePromise<U>.

template <typename T> struct UnwrapPromise_;
template <typename T> struct UnwrapPromise_<Promise<T>> { typedef T Type; };

template <typename T>
using UnwrapPromise = typename UnwrapPromise_<T>::Type;

class PropagateException {
  // A functor which accepts a kj::Exception as a parameter and returns a broken promise of
  // arbitrary type which simply propagates the exception.
public:
  class Bottom {
  public:
    Bottom(Exception&& exception): exception(kj::mv(exception)) {}

    Exception asException() { return kj::mv(exception); }

  private:
    Exception exception;
  };

  Bottom operator()(Exception&& e) {
    return Bottom(kj::mv(e));
  }
  Bottom operator()(const  Exception& e) {
    return Bottom(kj::cp(e));
  }
};

template <typename Func, typename T>
struct ReturnType_ { typedef decltype(instance<Func>()(instance<T>())) Type; };
template <typename Func>
struct ReturnType_<Func, void> { typedef decltype(instance<Func>()()) Type; };

template <typename Func, typename T>
using ReturnType = typename ReturnType_<Func, T>::Type;
// The return type of functor Func given a parameter of type T, with the special exception that if
// T is void, this is the return type of Func called with no arguments.

template <typename T> struct SplitTuplePromise_ { typedef Promise<T> Type; };
template <typename... T>
struct SplitTuplePromise_<kj::_::Tuple<T...>> {
  typedef kj::Tuple<ReducePromises<T>...> Type;
};

template <typename T>
using SplitTuplePromise = typename SplitTuplePromise_<T>::Type;
// T -> Promise<T>
// Tuple<T> -> Tuple<Promise<T>>

struct Void {};
// Application code should NOT refer to this!  See `kj::READY_NOW` instead.

template <typename T> struct FixVoid_ { typedef T Type; };
template <> struct FixVoid_<void> { typedef Void Type; };
template <typename T> using FixVoid = typename FixVoid_<T>::Type;
// FixVoid<T> is just T unless T is void in which case it is _::Void (an empty struct).

template <typename T> struct UnfixVoid_ { typedef T Type; };
template <> struct UnfixVoid_<Void> { typedef void Type; };
template <typename T> using UnfixVoid = typename UnfixVoid_<T>::Type;
// UnfixVoid is the opposite of FixVoid.

template <typename In, typename Out>
struct MaybeVoidCaller {
  // Calls the function converting a Void input to an empty parameter list and a void return
  // value to a Void output.

  template <typename Func>
  static inline Out apply(Func& func, In&& in) {
    return func(kj::mv(in));
  }
};
template <typename In, typename Out>
struct MaybeVoidCaller<In&, Out> {
  template <typename Func>
  static inline Out apply(Func& func, In& in) {
    return func(in);
  }
};
template <typename Out>
struct MaybeVoidCaller<Void, Out> {
  template <typename Func>
  static inline Out apply(Func& func, Void&& in) {
    return func();
  }
};
template <typename In>
struct MaybeVoidCaller<In, Void> {
  template <typename Func>
  static inline Void apply(Func& func, In&& in) {
    func(kj::mv(in));
    return Void();
  }
};
template <typename In>
struct MaybeVoidCaller<In&, Void> {
  template <typename Func>
  static inline Void apply(Func& func, In& in) {
    func(in);
    return Void();
  }
};
template <>
struct MaybeVoidCaller<Void, Void> {
  template <typename Func>
  static inline Void apply(Func& func, Void&& in) {
    func();
    return Void();
  }
};

template <typename T>
inline T&& returnMaybeVoid(T&& t) {
  return kj::fwd<T>(t);
}
inline void returnMaybeVoid(Void&& v) {}

class ExceptionOrValue;
class PromiseNode;
class ChainPromiseNode;
template <typename T>
class ForkHub;
class FiberStack;
class FiberBase;

class Event;
class XThreadEvent;
class XThreadPaf;

class PromiseDisposer;
using OwnPromiseNode = Own<PromiseNode, PromiseDisposer>;
// PromiseNode uses a static disposer.

class PromiseBase {
public:
  kj::String trace();
  // Dump debug info about this promise.

private:
  OwnPromiseNode node;

  PromiseBase() = default;
  PromiseBase(OwnPromiseNode&& node): node(kj::mv(node)) {}

  template <typename>
  friend class kj::Promise;
  friend class PromiseNode;
};

void detach(kj::Promise<void>&& promise);
void waitImpl(_::OwnPromiseNode&& node, _::ExceptionOrValue& result, WaitScope& waitScope,
              SourceLocation location);
bool pollImpl(_::PromiseNode& node, WaitScope& waitScope, SourceLocation location);
Promise<void> yield();
Promise<void> yieldHarder();
OwnPromiseNode readyNow();
OwnPromiseNode neverDone();

class ReadyNow {
public:
  operator Promise<void>() const;
};

class NeverDone {
public:
  template <typename T>
  operator Promise<T>() const;

  KJ_NORETURN(void wait(WaitScope& waitScope, SourceLocation location = {}) const);
};

}  // namespace _ (private)
}  // namespace kj

KJ_END_HEADER
