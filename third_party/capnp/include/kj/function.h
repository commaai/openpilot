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

#include "memory.h"

KJ_BEGIN_HEADER

namespace kj {

template <typename Signature>
class Function;
// Function wrapper using virtual-based polymorphism.  Use this when template polymorphism is
// not possible.  You can, for example, accept a Function as a parameter:
//
//     void setFilter(Function<bool(const Widget&)> filter);
//
// The caller of `setFilter()` may then pass any callable object as the parameter.  The callable
// object does not have to have the exact signature specified, just one that is "compatible" --
// i.e. the return type is covariant and the parameters are contravariant.
//
// Unlike `std::function`, `kj::Function`s are movable but not copyable, just like `kj::Own`.  This
// is to avoid unexpected heap allocation or slow atomic reference counting.
//
// When a `Function` is constructed from an lvalue, it captures only a reference to the value.
// When constructed from an rvalue, it invokes the value's move constructor.  So, for example:
//
//     struct AddN {
//       int n;
//       int operator(int i) { return i + n; }
//     }
//
//     Function<int(int, int)> f1 = AddN{2};
//     // f1 owns an instance of AddN.  It may safely be moved out
//     // of the local scope.
//
//     AddN adder(2);
//     Function<int(int, int)> f2 = adder;
//     // f2 contains a reference to `adder`.  Thus, it becomes invalid
//     // when `adder` goes out-of-scope.
//
//     AddN adder2(2);
//     Function<int(int, int)> f3 = kj::mv(adder2);
//     // f3 owns an insatnce of AddN moved from `adder2`.  f3 may safely
//     // be moved out of the local scope.
//
// Additionally, a Function may be bound to a class method using KJ_BIND_METHOD(object, methodName).
// For example:
//
//     class Printer {
//     public:
//       void print(int i);
//       void print(kj::StringPtr s);
//     };
//
//     Printer p;
//
//     Function<void(uint)> intPrinter = KJ_BIND_METHOD(p, print);
//     // Will call Printer::print(int).
//
//     Function<void(const char*)> strPrinter = KJ_BIND_METHOD(p, print);
//     // Will call Printer::print(kj::StringPtr).
//
// Notice how KJ_BIND_METHOD is able to figure out which overload to use depending on the kind of
// Function it is binding to.

template <typename Signature>
class ConstFunction;
// Like Function, but wraps a "const" (i.e. thread-safe) call.

template <typename Signature>
class FunctionParam;
// Like Function, but used specifically as a call parameter type. Does not do any heap allocation.
//
// This type MUST NOT be used for anything other than a parameter type to a function or method.
// This is because if FunctionParam binds to a temporary, it assumes that the temporary will
// outlive the FunctionParam instance. This is true when FunctionParam is used as a parameter type,
// but not if it is used as a local variable nor a class member variable.

template <typename Return, typename... Params>
class Function<Return(Params...)> {
public:
  template <typename F>
  inline Function(F&& f): impl(heap<Impl<F>>(kj::fwd<F>(f))) {}
  Function() = default;

  // Make sure people don't accidentally end up wrapping a reference when they meant to return
  // a function.
  KJ_DISALLOW_COPY(Function);
  Function(Function&) = delete;
  Function& operator=(Function&) = delete;
  template <typename T> Function(const Function<T>&) = delete;
  template <typename T> Function& operator=(const Function<T>&) = delete;
  template <typename T> Function(const ConstFunction<T>&) = delete;
  template <typename T> Function& operator=(const ConstFunction<T>&) = delete;
  Function(Function&&) = default;
  Function& operator=(Function&&) = default;

  inline Return operator()(Params... params) {
    return (*impl)(kj::fwd<Params>(params)...);
  }

  Function reference() {
    // Forms a new Function of the same type that delegates to this Function by reference.
    // Therefore, this Function must outlive the returned Function, but otherwise they behave
    // exactly the same.

    return *impl;
  }

private:
  class Iface {
  public:
    virtual Return operator()(Params... params) = 0;
  };

  template <typename F>
  class Impl final: public Iface {
  public:
    explicit Impl(F&& f): f(kj::fwd<F>(f)) {}

    Return operator()(Params... params) override {
      return f(kj::fwd<Params>(params)...);
    }

  private:
    F f;
  };

  Own<Iface> impl;
};

template <typename Return, typename... Params>
class ConstFunction<Return(Params...)> {
public:
  template <typename F>
  inline ConstFunction(F&& f): impl(heap<Impl<F>>(kj::fwd<F>(f))) {}
  ConstFunction() = default;

  // Make sure people don't accidentally end up wrapping a reference when they meant to return
  // a function.
  KJ_DISALLOW_COPY(ConstFunction);
  ConstFunction(ConstFunction&) = delete;
  ConstFunction& operator=(ConstFunction&) = delete;
  template <typename T> ConstFunction(const ConstFunction<T>&) = delete;
  template <typename T> ConstFunction& operator=(const ConstFunction<T>&) = delete;
  template <typename T> ConstFunction(const Function<T>&) = delete;
  template <typename T> ConstFunction& operator=(const Function<T>&) = delete;
  ConstFunction(ConstFunction&&) = default;
  ConstFunction& operator=(ConstFunction&&) = default;

  inline Return operator()(Params... params) const {
    return (*impl)(kj::fwd<Params>(params)...);
  }

  ConstFunction reference() const {
    // Forms a new ConstFunction of the same type that delegates to this ConstFunction by reference.
    // Therefore, this ConstFunction must outlive the returned ConstFunction, but otherwise they
    // behave exactly the same.

    return *impl;
  }

private:
  class Iface {
  public:
    virtual Return operator()(Params... params) const = 0;
  };

  template <typename F>
  class Impl final: public Iface {
  public:
    explicit Impl(F&& f): f(kj::fwd<F>(f)) {}

    Return operator()(Params... params) const override {
      return f(kj::fwd<Params>(params)...);
    }

  private:
    F f;
  };

  Own<Iface> impl;
};

template <typename Return, typename... Params>
class FunctionParam<Return(Params...)> {
public:
  template <typename Func>
  FunctionParam(Func&& func) {
    typedef Wrapper<Decay<Func>> WrapperType;

    // All instances of Wrapper<Func> are two pointers in size: a vtable, and a Func&. So if we
    // allocate space for two pointers, we can construct a Wrapper<Func> in it!
    static_assert(sizeof(WrapperType) == sizeof(space),
        "expected WrapperType to be two pointers");

    // Even if `func` is an rvalue reference, it's OK to use it as an lvalue here, because
    // FunctionParam is used strictly for parameters. If we captured a temporary, we know that
    // temporary will not be destroyed until after the function call completes.
    ctor(*reinterpret_cast<WrapperType*>(space), func);
  }

  FunctionParam(const FunctionParam& other) = default;
  FunctionParam(FunctionParam&& other) = default;
  // Magically, a plain copy works.

  inline Return operator()(Params... params) {
    return (*reinterpret_cast<WrapperBase*>(space))(kj::fwd<Params>(params)...);
  }

private:
  alignas(void*) char space[2 * sizeof(void*)];

  class WrapperBase {
  public:
    virtual Return operator()(Params... params) = 0;
  };

  template <typename Func>
  class Wrapper: public WrapperBase {
  public:
    Wrapper(Func& func): func(func) {}

    inline Return operator()(Params... params) override {
      return func(kj::fwd<Params>(params)...);
    }

  private:
    Func& func;
  };
};

namespace _ {  // private

template <typename T, typename Func, typename ConstFunc>
class BoundMethod {
public:
  BoundMethod(T&& t, Func&& func, ConstFunc&& constFunc)
      : t(kj::fwd<T>(t)), func(kj::mv(func)), constFunc(kj::mv(constFunc)) {}

  template <typename... Params>
  auto operator()(Params&&... params) {
    return func(t, kj::fwd<Params>(params)...);
  }
  template <typename... Params>
  auto operator()(Params&&... params) const {
    return constFunc(t, kj::fwd<Params>(params)...);
  }

private:
  T t;
  Func func;
  ConstFunc constFunc;
};

template <typename T, typename Func, typename ConstFunc>
BoundMethod<T, Func, ConstFunc> boundMethod(T&& t, Func&& func, ConstFunc&& constFunc) {
  return { kj::fwd<T>(t), kj::fwd<Func>(func), kj::fwd<ConstFunc>(constFunc) };
}

}  // namespace _ (private)

#define KJ_BIND_METHOD(obj, method) \
  ::kj::_::boundMethod(obj, \
      [](auto& s, auto&&... p) mutable { return s.method(kj::fwd<decltype(p)>(p)...); }, \
      [](auto& s, auto&&... p) { return s.method(kj::fwd<decltype(p)>(p)...); })
// Macro that produces a functor object which forwards to the method `obj.name`.  If `obj` is an
// lvalue, the functor will hold a reference to it.  If `obj` is an rvalue, the functor will
// contain a copy (by move) of it. The method is allowed to be overloaded.

}  // namespace kj

KJ_END_HEADER
