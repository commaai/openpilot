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

// This file defines a notion of tuples that is simpler than `std::tuple`.  It works as follows:
// - `kj::Tuple<A, B, C> is the type of a tuple of an A, a B, and a C.
// - `kj::tuple(a, b, c)` returns a tuple containing a, b, and c.  If any of these are themselves
//   tuples, they are flattened, so `tuple(a, tuple(b, c), d)` is equivalent to `tuple(a, b, c, d)`.
// - `kj::get<n>(myTuple)` returns the element of `myTuple` at index n.
// - `kj::apply(func, ...)` calls func on the following arguments after first expanding any tuples
//   in the argument list.  So `kj::apply(foo, a, tuple(b, c), d)` would call `foo(a, b, c, d)`.
//
// Note that:
// - The type `Tuple<T>` is a synonym for T.  This is why `get` and `apply` are not members of the
//   type.
// - It is illegal for an element of `Tuple` to itself be a tuple, as tuples are meant to be
//   flattened.
// - It is illegal for an element of `Tuple` to be a reference, due to problems this would cause
//   with type inference and `tuple()`.

#pragma once

#include "common.h"

KJ_BEGIN_HEADER

namespace kj {
namespace _ {  // private

template <size_t index, typename... T>
struct TypeByIndex_;
template <typename First, typename... Rest>
struct TypeByIndex_<0, First, Rest...> {
  typedef First Type;
};
template <size_t index, typename First, typename... Rest>
struct TypeByIndex_<index, First, Rest...>
    : public TypeByIndex_<index - 1, Rest...> {};
template <size_t index>
struct TypeByIndex_<index> {
  static_assert(index != index, "Index out-of-range.");
};
template <size_t index, typename... T>
using TypeByIndex = typename TypeByIndex_<index, T...>::Type;
// Chose a particular type out of a list of types, by index.

template <size_t... s>
struct Indexes {};
// Dummy helper type that just encapsulates a sequential list of indexes, so that we can match
// templates against them and unpack them with '...'.

template <size_t end, size_t... prefix>
struct MakeIndexes_: public MakeIndexes_<end - 1, end - 1, prefix...> {};
template <size_t... prefix>
struct MakeIndexes_<0, prefix...> {
  typedef Indexes<prefix...> Type;
};
template <size_t end>
using MakeIndexes = typename MakeIndexes_<end>::Type;
// Equivalent to Indexes<0, 1, 2, ..., end>.

template <typename... T>
class Tuple;
template <size_t index, typename... U>
inline TypeByIndex<index, U...>& getImpl(Tuple<U...>& tuple);
template <size_t index, typename... U>
inline TypeByIndex<index, U...>&& getImpl(Tuple<U...>&& tuple);
template <size_t index, typename... U>
inline const TypeByIndex<index, U...>& getImpl(const Tuple<U...>& tuple);

template <uint index, typename T>
struct TupleElement {
  // Encapsulates one element of a tuple.  The actual tuple implementation multiply-inherits
  // from a TupleElement for each element, which is more efficient than a recursive definition.

  T value;
  TupleElement() = default;
  constexpr inline TupleElement(const T& value): value(value) {}
  constexpr inline TupleElement(T&& value): value(kj::mv(value)) {}
};

template <uint index, typename T>
struct TupleElement<index, T&> {
  // A tuple containing references can be constructed using refTuple().

  T& value;
  constexpr inline TupleElement(T& value): value(value) {}
};

template <uint index, typename... T>
struct TupleElement<index, Tuple<T...>> {
  static_assert(sizeof(Tuple<T...>*) == 0,
                "Tuples cannot contain other tuples -- they should be flattened.");
};

template <typename Indexes, typename... Types>
struct TupleImpl;

template <size_t... indexes, typename... Types>
struct TupleImpl<Indexes<indexes...>, Types...>
    : public TupleElement<indexes, Types>... {
  // Implementation of Tuple.  The only reason we need this rather than rolling this into class
  // Tuple (below) is so that we can get "indexes" as an unpackable list.

  static_assert(sizeof...(indexes) == sizeof...(Types), "Incorrect use of TupleImpl.");

  TupleImpl() = default;

  template <typename... Params>
  inline TupleImpl(Params&&... params)
      : TupleElement<indexes, Types>(kj::fwd<Params>(params))... {
    // Work around Clang 3.2 bug 16303 where this is not detected.  (Unfortunately, Clang sometimes
    // segfaults instead.)
    static_assert(sizeof...(params) == sizeof...(indexes),
                  "Wrong number of parameters to Tuple constructor.");
  }

  template <typename... U>
  constexpr inline TupleImpl(Tuple<U...>&& other)
      : TupleElement<indexes, Types>(kj::fwd<U>(getImpl<indexes>(other)))... {}
  template <typename... U>
  constexpr inline TupleImpl(Tuple<U...>& other)
      : TupleElement<indexes, Types>(getImpl<indexes>(other))... {}
  template <typename... U>
  constexpr inline TupleImpl(const Tuple<U...>& other)
      : TupleElement<indexes, Types>(getImpl<indexes>(other))... {}
};

struct MakeTupleFunc;
struct MakeRefTupleFunc;

template <typename... T>
class Tuple {
  // The actual Tuple class (used for tuples of size other than 1).

public:
  Tuple() = default;

  template <typename... U>
  constexpr inline Tuple(Tuple<U...>&& other): impl(kj::mv(other)) {}
  template <typename... U>
  constexpr inline Tuple(Tuple<U...>& other): impl(other) {}
  template <typename... U>
  constexpr inline Tuple(const Tuple<U...>& other): impl(other) {}

private:
  template <typename... Params>
  constexpr Tuple(Params&&... params): impl(kj::fwd<Params>(params)...) {}

  TupleImpl<MakeIndexes<sizeof...(T)>, T...> impl;

  template <size_t index, typename... U>
  friend inline TypeByIndex<index, U...>& getImpl(Tuple<U...>& tuple);
  template <size_t index, typename... U>
  friend inline TypeByIndex<index, U...>&& getImpl(Tuple<U...>&& tuple);
  template <size_t index, typename... U>
  friend inline const TypeByIndex<index, U...>& getImpl(const Tuple<U...>& tuple);
  friend struct MakeTupleFunc;
  friend struct MakeRefTupleFunc;
};

template <>
class Tuple<> {
  // Simplified zero-member version of Tuple.  In particular this is important to make sure that
  // Tuple<>() is constexpr.
};

template <typename T>
class Tuple<T>;
// Single-element tuple should never be used.  The public API should ensure this.

template <size_t index, typename... T>
inline TypeByIndex<index, T...>& getImpl(Tuple<T...>& tuple) {
  // Get member of a Tuple by index, e.g. `get<2>(myTuple)`.
  static_assert(index < sizeof...(T), "Tuple element index out-of-bounds.");
  return implicitCast<TupleElement<index, TypeByIndex<index, T...>>&>(tuple.impl).value;
}
template <size_t index, typename... T>
inline TypeByIndex<index, T...>&& getImpl(Tuple<T...>&& tuple) {
  // Get member of a Tuple by index, e.g. `get<2>(myTuple)`.
  static_assert(index < sizeof...(T), "Tuple element index out-of-bounds.");
  return kj::mv(implicitCast<TupleElement<index, TypeByIndex<index, T...>>&>(tuple.impl).value);
}
template <size_t index, typename... T>
inline const TypeByIndex<index, T...>& getImpl(const Tuple<T...>& tuple) {
  // Get member of a Tuple by index, e.g. `get<2>(myTuple)`.
  static_assert(index < sizeof...(T), "Tuple element index out-of-bounds.");
  return implicitCast<const TupleElement<index, TypeByIndex<index, T...>>&>(tuple.impl).value;
}
template <size_t index, typename T>
inline T&& getImpl(T&& value) {
  // Get member of a Tuple by index, e.g. `getImpl<2>(myTuple)`.

  // Non-tuples are equivalent to one-element tuples.
  static_assert(index == 0, "Tuple element index out-of-bounds.");
  return kj::fwd<T>(value);
}


template <typename Func, typename SoFar, typename... T>
struct ExpandAndApplyResult_;
// Template which computes the return type of applying Func to T... after flattening tuples.
// SoFar starts as Tuple<> and accumulates the flattened parameter types -- so after this template
// is recursively expanded, T... is empty and SoFar is a Tuple containing all the parameters.

template <typename Func, typename First, typename... Rest, typename... T>
struct ExpandAndApplyResult_<Func, Tuple<T...>, First, Rest...>
    : public ExpandAndApplyResult_<Func, Tuple<T..., First>, Rest...> {};
template <typename Func, typename... FirstTypes, typename... Rest, typename... T>
struct ExpandAndApplyResult_<Func, Tuple<T...>, Tuple<FirstTypes...>, Rest...>
    : public ExpandAndApplyResult_<Func, Tuple<T...>, FirstTypes&&..., Rest...> {};
template <typename Func, typename... FirstTypes, typename... Rest, typename... T>
struct ExpandAndApplyResult_<Func, Tuple<T...>, Tuple<FirstTypes...>&, Rest...>
    : public ExpandAndApplyResult_<Func, Tuple<T...>, FirstTypes&..., Rest...> {};
template <typename Func, typename... FirstTypes, typename... Rest, typename... T>
struct ExpandAndApplyResult_<Func, Tuple<T...>, const Tuple<FirstTypes...>&, Rest...>
    : public ExpandAndApplyResult_<Func, Tuple<T...>, const FirstTypes&..., Rest...> {};
template <typename Func, typename... T>
struct ExpandAndApplyResult_<Func, Tuple<T...>> {
  typedef decltype(instance<Func>()(instance<T&&>()...)) Type;
};
template <typename Func, typename... T>
using ExpandAndApplyResult = typename ExpandAndApplyResult_<Func, Tuple<>, T...>::Type;
// Computes the expected return type of `expandAndApply()`.

template <typename Func>
inline auto expandAndApply(Func&& func) -> ExpandAndApplyResult<Func> {
  return func();
}

template <typename Func, typename First, typename... Rest>
struct ExpandAndApplyFunc {
  Func&& func;
  First&& first;
  ExpandAndApplyFunc(Func&& func, First&& first)
      : func(kj::fwd<Func>(func)), first(kj::fwd<First>(first)) {}
  template <typename... T>
  auto operator()(T&&... params)
      -> decltype(this->func(kj::fwd<First>(first), kj::fwd<T>(params)...)) {
    return this->func(kj::fwd<First>(first), kj::fwd<T>(params)...);
  }
};

template <typename Func, typename First, typename... Rest>
inline auto expandAndApply(Func&& func, First&& first, Rest&&... rest)
    -> ExpandAndApplyResult<Func, First, Rest...> {

  return expandAndApply(
      ExpandAndApplyFunc<Func, First, Rest...>(kj::fwd<Func>(func), kj::fwd<First>(first)),
      kj::fwd<Rest>(rest)...);
}

template <typename Func, typename... FirstTypes, typename... Rest>
inline auto expandAndApply(Func&& func, Tuple<FirstTypes...>&& first, Rest&&... rest)
    -> ExpandAndApplyResult<Func, FirstTypes&&..., Rest...> {
  return expandAndApplyWithIndexes(MakeIndexes<sizeof...(FirstTypes)>(),
      kj::fwd<Func>(func), kj::mv(first), kj::fwd<Rest>(rest)...);
}

template <typename Func, typename... FirstTypes, typename... Rest>
inline auto expandAndApply(Func&& func, Tuple<FirstTypes...>& first, Rest&&... rest)
    -> ExpandAndApplyResult<Func, FirstTypes..., Rest...> {
  return expandAndApplyWithIndexes(MakeIndexes<sizeof...(FirstTypes)>(),
      kj::fwd<Func>(func), first, kj::fwd<Rest>(rest)...);
}

template <typename Func, typename... FirstTypes, typename... Rest>
inline auto expandAndApply(Func&& func, const Tuple<FirstTypes...>& first, Rest&&... rest)
    -> ExpandAndApplyResult<Func, FirstTypes..., Rest...> {
  return expandAndApplyWithIndexes(MakeIndexes<sizeof...(FirstTypes)>(),
      kj::fwd<Func>(func), first, kj::fwd<Rest>(rest)...);
}

template <typename Func, typename... FirstTypes, typename... Rest, size_t... indexes>
inline auto expandAndApplyWithIndexes(
    Indexes<indexes...>, Func&& func, Tuple<FirstTypes...>&& first, Rest&&... rest)
    -> ExpandAndApplyResult<Func, FirstTypes&&..., Rest...> {
  return expandAndApply(kj::fwd<Func>(func), kj::mv(getImpl<indexes>(first))...,
                        kj::fwd<Rest>(rest)...);
}

template <typename Func, typename... FirstTypes, typename... Rest, size_t... indexes>
inline auto expandAndApplyWithIndexes(
    Indexes<indexes...>, Func&& func, const Tuple<FirstTypes...>& first, Rest&&... rest)
    -> ExpandAndApplyResult<Func, FirstTypes..., Rest...> {
  return expandAndApply(kj::fwd<Func>(func), getImpl<indexes>(first)...,
                       kj::fwd<Rest>(rest)...);
}

struct MakeTupleFunc {
  template <typename... Params>
  Tuple<Decay<Params>...> operator()(Params&&... params) {
    return Tuple<Decay<Params>...>(kj::fwd<Params>(params)...);
  }
  template <typename Param>
  Decay<Param> operator()(Param&& param) {
    return kj::fwd<Param>(param);
  }
};

struct MakeRefTupleFunc {
  template <typename... Params>
  Tuple<Params...> operator()(Params&&... params) {
    return Tuple<Params...>(kj::fwd<Params>(params)...);
  }
  template <typename Param>
  Param operator()(Param&& param) {
    return kj::fwd<Param>(param);
  }
};

}  // namespace _ (private)

template <typename... T> struct Tuple_ { typedef _::Tuple<T...> Type; };
template <typename T> struct Tuple_<T> { typedef T Type; };

template <typename... T> using Tuple = typename Tuple_<T...>::Type;
// Tuple type.  `Tuple<T>` (i.e. a single-element tuple) is a synonym for `T`.  Tuples of size
// other than 1 expand to an internal type.  Either way, you can construct a Tuple using
// `kj::tuple(...)`, get an element by index `i` using `kj::get<i>(myTuple)`, and expand the tuple
// as arguments to a function using `kj::apply(func, myTuple)`.
//
// Tuples are always flat -- that is, no element of a Tuple is ever itself a Tuple.  If you
// construct a tuple from other tuples, the elements are flattened and concatenated.

template <typename... Params>
inline auto tuple(Params&&... params)
    -> decltype(_::expandAndApply(_::MakeTupleFunc(), kj::fwd<Params>(params)...)) {
  // Construct a new tuple from the given values.  Any tuples in the argument list will be
  // flattened into the result.
  return _::expandAndApply(_::MakeTupleFunc(), kj::fwd<Params>(params)...);
}

template <typename... Params>
inline auto refTuple(Params&&... params)
    -> decltype(_::expandAndApply(_::MakeRefTupleFunc(), kj::fwd<Params>(params)...)) {
  // Like tuple(), but if the params include lvalue references, they will be captured as
  // references. rvalue references will still be captured as whole values (moved).
  return _::expandAndApply(_::MakeRefTupleFunc(), kj::fwd<Params>(params)...);
}

template <size_t index, typename Tuple>
inline auto get(Tuple&& tuple) -> decltype(_::getImpl<index>(kj::fwd<Tuple>(tuple))) {
  // Unpack and return the tuple element at the given index.  The index is specified as a template
  // parameter, e.g. `kj::get<3>(myTuple)`.
  return _::getImpl<index>(kj::fwd<Tuple>(tuple));
}

template <typename Func, typename... Params>
inline auto apply(Func&& func, Params&&... params)
    -> decltype(_::expandAndApply(kj::fwd<Func>(func), kj::fwd<Params>(params)...)) {
  // Apply a function to some arguments, expanding tuples into separate arguments.
  return _::expandAndApply(kj::fwd<Func>(func), kj::fwd<Params>(params)...);
}

template <typename T> struct TupleSize_ { static constexpr size_t size = 1; };
template <typename... T> struct TupleSize_<_::Tuple<T...>> {
  static constexpr size_t size = sizeof...(T);
};

template <typename T>
constexpr size_t tupleSize() { return TupleSize_<T>::size; }
// Returns size of the tuple T.

template <typename T, typename Tuple>
struct IndexOfType_;
template <typename T, typename Tuple>
struct HasType_ {
  static constexpr bool value = false;
};

template <typename T>
struct IndexOfType_<T, T> {
  static constexpr size_t value = 0;
};
template <typename T>
struct HasType_<T, T> {
  static constexpr bool value = true;
};

template <typename T, typename... U>
struct IndexOfType_<T, _::Tuple<T, U...>> {
  static constexpr size_t value = 0;
  static_assert(!HasType_<T, _::Tuple<U...>>::value,
      "requested type appears multiple times in tuple");
};
template <typename T, typename... U>
struct HasType_<T, _::Tuple<T, U...>> {
  static constexpr bool value = true;
};

template <typename T, typename U, typename... V>
struct IndexOfType_<T, _::Tuple<U, V...>> {
  static constexpr size_t value = IndexOfType_<T, _::Tuple<V...>>::value + 1;
};
template <typename T, typename U, typename... V>
struct HasType_<T, _::Tuple<U, V...>> {
  static constexpr bool value = HasType_<T, _::Tuple<V...>>::value;
};

template <typename T, typename U>
inline constexpr size_t indexOfType() {
  static_assert(HasType_<T, U>::value, "type not present");
  return IndexOfType_<T, U>::value;
}

template <size_t i, typename T>
struct TypeOfIndex_;
template <typename T>
struct TypeOfIndex_<0, T> {
  typedef T Type;
};
template <size_t i, typename T, typename... U>
struct TypeOfIndex_<i, _::Tuple<T, U...>>
    : public TypeOfIndex_<i - 1, _::Tuple<U...>> {};
template <typename T, typename... U>
struct TypeOfIndex_<0, _::Tuple<T, U...>> {
  typedef T Type;
};

template <size_t i, typename Tuple>
using TypeOfIndex = typename TypeOfIndex_<i, Tuple>::Type;

}  // namespace kj

KJ_END_HEADER
