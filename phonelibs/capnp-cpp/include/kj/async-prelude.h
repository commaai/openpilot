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

#ifndef KJ_ASYNC_PRELUDE_H_
#define KJ_ASYNC_PRELUDE_H_

#if defined(__GNUC__) && !KJ_HEADER_WARNINGS
#pragma GCC system_header
#endif

#include "exception.h"
#include "tuple.h"

namespace kj {

class EventLoop;
template <typename T>
class Promise;
class WaitScope;

template <typename T>
Promise<Array<T>> joinPromises(Array<Promise<T>>&& promises);
Promise<void> joinPromises(Array<Promise<void>>&& promises);

namespace _ {  // private

template <typename T> struct JoinPromises_ { typedef T Type; };
template <typename T> struct JoinPromises_<Promise<T>> { typedef T Type; };

template <typename T>
using JoinPromises = typename JoinPromises_<T>::Type;
// If T is Promise<U>, resolves to U, otherwise resolves to T.
//
// TODO(cleanup):  Rename to avoid confusion with joinPromises() call which is completely
//   unrelated.

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
  typedef kj::Tuple<Promise<JoinPromises<T>>...> Type;
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

class TaskSetImpl;

class Event;

class PromiseBase {
public:
  kj::String trace();
  // Dump debug info about this promise.

private:
  Own<PromiseNode> node;

  PromiseBase() = default;
  PromiseBase(Own<PromiseNode>&& node): node(kj::mv(node)) {}

  friend class kj::EventLoop;
  friend class ChainPromiseNode;
  template <typename>
  friend class kj::Promise;
  friend class TaskSetImpl;
  template <typename U>
  friend Promise<Array<U>> kj::joinPromises(Array<Promise<U>>&& promises);
  friend Promise<void> kj::joinPromises(Array<Promise<void>>&& promises);
};

void detach(kj::Promise<void>&& promise);
void waitImpl(Own<_::PromiseNode>&& node, _::ExceptionOrValue& result, WaitScope& waitScope);
Promise<void> yield();
Own<PromiseNode> neverDone();

class NeverDone {
public:
  template <typename T>
  operator Promise<T>() const {
    return Promise<T>(false, neverDone());
  }

  KJ_NORETURN(void wait(WaitScope& waitScope) const);
};

}  // namespace _ (private)
}  // namespace kj

#endif  // KJ_ASYNC_PRELUDE_H_
