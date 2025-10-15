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

// This file contains types which are intended to help detect incorrect usage at compile
// time, but should then be optimized down to basic primitives (usually, integers) by the
// compiler.

#pragma once

#include "common.h"
#include <inttypes.h>

KJ_BEGIN_HEADER

namespace kj {

// =======================================================================================
// IDs

template <typename UnderlyingType, typename Label>
struct Id {
  // A type-safe numeric ID.  `UnderlyingType` is the underlying integer representation.  `Label`
  // distinguishes this Id from other Id types.  Sample usage:
  //
  //   class Foo;
  //   typedef Id<uint, Foo> FooId;
  //
  //   class Bar;
  //   typedef Id<uint, Bar> BarId;
  //
  // You can now use the FooId and BarId types without any possibility of accidentally using a
  // FooId when you really wanted a BarId or vice-versa.

  UnderlyingType value;

  inline constexpr Id(): value(0) {}
  inline constexpr explicit Id(int value): value(value) {}

  inline constexpr bool operator==(const Id& other) const { return value == other.value; }
  inline constexpr bool operator!=(const Id& other) const { return value != other.value; }
  inline constexpr bool operator<=(const Id& other) const { return value <= other.value; }
  inline constexpr bool operator>=(const Id& other) const { return value >= other.value; }
  inline constexpr bool operator< (const Id& other) const { return value <  other.value; }
  inline constexpr bool operator> (const Id& other) const { return value >  other.value; }
};

// =======================================================================================
// Quantity and UnitRatio -- implement unit analysis via the type system

struct Unsafe_ {};
constexpr Unsafe_ unsafe = Unsafe_();
// Use as a parameter to constructors that are unsafe to indicate that you really do mean it.

template <uint64_t maxN, typename T>
class Bounded;
template <uint value>
class BoundedConst;

template <typename T>
struct IsIntegralOrBounded_ { static constexpr bool value = isIntegral<T>(); };
template <uint64_t m, typename T>
struct IsIntegralOrBounded_<Bounded<m, T>> { static constexpr bool value = true; };
template <uint v>
struct IsIntegralOrBounded_<BoundedConst<v>> { static constexpr bool value = true; };

template <typename T>
inline constexpr bool isIntegralOrBounded() { return IsIntegralOrBounded_<T>::value; }

template <typename Number, typename Unit1, typename Unit2>
class UnitRatio {
  // A multiplier used to convert Quantities of one unit to Quantities of another unit.  See
  // Quantity, below.
  //
  // Construct this type by dividing one Quantity by another of a different unit.  Use this type
  // by multiplying it by a Quantity, or dividing a Quantity by it.

  static_assert(isIntegralOrBounded<Number>(),
      "Underlying type for UnitRatio must be integer.");

public:
  inline UnitRatio() {}

  constexpr UnitRatio(Number unit1PerUnit2, decltype(unsafe)): unit1PerUnit2(unit1PerUnit2) {}
  // This constructor was intended to be private, but GCC complains about it being private in a
  // bunch of places that don't appear to even call it, so I made it public.  Oh well.

  template <typename OtherNumber>
  inline constexpr UnitRatio(const UnitRatio<OtherNumber, Unit1, Unit2>& other)
      : unit1PerUnit2(other.unit1PerUnit2) {}

  template <typename OtherNumber>
  inline constexpr UnitRatio<decltype(Number()+OtherNumber()), Unit1, Unit2>
      operator+(UnitRatio<OtherNumber, Unit1, Unit2> other) const {
    return UnitRatio<decltype(Number()+OtherNumber()), Unit1, Unit2>(
        unit1PerUnit2 + other.unit1PerUnit2, unsafe);
  }
  template <typename OtherNumber>
  inline constexpr UnitRatio<decltype(Number()-OtherNumber()), Unit1, Unit2>
      operator-(UnitRatio<OtherNumber, Unit1, Unit2> other) const {
    return UnitRatio<decltype(Number()-OtherNumber()), Unit1, Unit2>(
        unit1PerUnit2 - other.unit1PerUnit2, unsafe);
  }

  template <typename OtherNumber, typename Unit3>
  inline constexpr UnitRatio<decltype(Number()*OtherNumber()), Unit3, Unit2>
      operator*(UnitRatio<OtherNumber, Unit3, Unit1> other) const {
    // U1 / U2 * U3 / U1 = U3 / U2
    return UnitRatio<decltype(Number()*OtherNumber()), Unit3, Unit2>(
        unit1PerUnit2 * other.unit1PerUnit2, unsafe);
  }
  template <typename OtherNumber, typename Unit3>
  inline constexpr UnitRatio<decltype(Number()*OtherNumber()), Unit1, Unit3>
      operator*(UnitRatio<OtherNumber, Unit2, Unit3> other) const {
    // U1 / U2 * U2 / U3 = U1 / U3
    return UnitRatio<decltype(Number()*OtherNumber()), Unit1, Unit3>(
        unit1PerUnit2 * other.unit1PerUnit2, unsafe);
  }

  template <typename OtherNumber, typename Unit3>
  inline constexpr UnitRatio<decltype(Number()*OtherNumber()), Unit3, Unit2>
      operator/(UnitRatio<OtherNumber, Unit1, Unit3> other) const {
    // (U1 / U2) / (U1 / U3) = U3 / U2
    return UnitRatio<decltype(Number()*OtherNumber()), Unit3, Unit2>(
        unit1PerUnit2 / other.unit1PerUnit2, unsafe);
  }
  template <typename OtherNumber, typename Unit3>
  inline constexpr UnitRatio<decltype(Number()*OtherNumber()), Unit1, Unit3>
      operator/(UnitRatio<OtherNumber, Unit3, Unit2> other) const {
    // (U1 / U2) / (U3 / U2) = U1 / U3
    return UnitRatio<decltype(Number()*OtherNumber()), Unit1, Unit3>(
        unit1PerUnit2 / other.unit1PerUnit2, unsafe);
  }

  template <typename OtherNumber>
  inline decltype(Number() / OtherNumber())
      operator/(UnitRatio<OtherNumber, Unit1, Unit2> other) const {
    return unit1PerUnit2 / other.unit1PerUnit2;
  }

  template <typename OtherNumber>
  inline constexpr bool operator==(const UnitRatio<OtherNumber, Unit1, Unit2>& other) const {
    return unit1PerUnit2 == other.unit1PerUnit2;
  }
  template <typename OtherNumber>
  inline constexpr bool operator!=(const UnitRatio<OtherNumber, Unit1, Unit2>& other) const {
    return unit1PerUnit2 != other.unit1PerUnit2;
  }

private:
  Number unit1PerUnit2;

  template <typename OtherNumber, typename OtherUnit>
  friend class Quantity;
  template <typename OtherNumber, typename OtherUnit1, typename OtherUnit2>
  friend class UnitRatio;

  template <typename N1, typename N2, typename U1, typename U2, typename>
  friend inline constexpr UnitRatio<decltype(N1() * N2()), U1, U2>
      operator*(N1, UnitRatio<N2, U1, U2>);
};

template <typename N1, typename N2, typename U1, typename U2,
          typename = EnableIf<isIntegralOrBounded<N1>() && isIntegralOrBounded<N2>()>>
inline constexpr UnitRatio<decltype(N1() * N2()), U1, U2>
    operator*(N1 n, UnitRatio<N2, U1, U2> r) {
  return UnitRatio<decltype(N1() * N2()), U1, U2>(n * r.unit1PerUnit2, unsafe);
}

template <typename Number, typename Unit>
class Quantity {
  // A type-safe numeric quantity, specified in terms of some unit.  Two Quantities cannot be used
  // in arithmetic unless they use the same unit.  The `Unit` type parameter is only used to prevent
  // accidental mixing of units; this type is never instantiated and can very well be incomplete.
  // `Number` is the underlying primitive numeric type.
  //
  // Quantities support most basic arithmetic operators, intelligently handling units, and
  // automatically casting the underlying type in the same way that the compiler would.
  //
  // To convert a primitive number to a Quantity, multiply it by unit<Quantity<N, U>>().
  // To convert a Quantity to a primitive number, divide it by unit<Quantity<N, U>>().
  // To convert a Quantity of one unit to another unit, multiply or divide by a UnitRatio.
  //
  // The Quantity class is not well-suited to hardcore physics as it does not allow multiplying
  // one quantity by another.  For example, multiplying meters by meters won't get you square
  // meters; it will get you a compiler error.  It would be interesting to see if template
  // metaprogramming could properly deal with such things but this isn't needed for the present
  // use case.
  //
  // Sample usage:
  //
  //   class SecondsLabel;
  //   typedef Quantity<double, SecondsLabel> Seconds;
  //   constexpr Seconds SECONDS = unit<Seconds>();
  //
  //   class MinutesLabel;
  //   typedef Quantity<double, MinutesLabel> Minutes;
  //   constexpr Minutes MINUTES = unit<Minutes>();
  //
  //   constexpr UnitRatio<double, SecondsLabel, MinutesLabel> SECONDS_PER_MINUTE =
  //       60 * SECONDS / MINUTES;
  //
  //   void waitFor(Seconds seconds) {
  //     sleep(seconds / SECONDS);
  //   }
  //   void waitFor(Minutes minutes) {
  //     waitFor(minutes * SECONDS_PER_MINUTE);
  //   }
  //
  //   void waitThreeMinutes() {
  //     waitFor(3 * MINUTES);
  //   }

  static_assert(isIntegralOrBounded<Number>(),
      "Underlying type for Quantity must be integer.");

public:
  inline constexpr Quantity() = default;

  inline constexpr Quantity(MaxValue_): value(maxValue) {}
  inline constexpr Quantity(MinValue_): value(minValue) {}
  // Allow initialization from maxValue and minValue.
  // TODO(msvc): decltype(maxValue) and decltype(minValue) deduce unknown-type for these function
  // parameters, causing the compiler to complain of a duplicate constructor definition, so we
  // specify MaxValue_ and MinValue_ types explicitly.

  inline constexpr Quantity(Number value, decltype(unsafe)): value(value) {}
  // This constructor was intended to be private, but GCC complains about it being private in a
  // bunch of places that don't appear to even call it, so I made it public.  Oh well.

  template <typename OtherNumber>
  inline constexpr Quantity(const Quantity<OtherNumber, Unit>& other)
      : value(other.value) {}

  template <typename OtherNumber>
  inline Quantity& operator=(const Quantity<OtherNumber, Unit>& other) {
    value = other.value;
    return *this;
  }

  template <typename OtherNumber>
  inline constexpr Quantity<decltype(Number() + OtherNumber()), Unit>
      operator+(const Quantity<OtherNumber, Unit>& other) const {
    return Quantity<decltype(Number() + OtherNumber()), Unit>(value + other.value, unsafe);
  }
  template <typename OtherNumber>
  inline constexpr Quantity<decltype(Number() - OtherNumber()), Unit>
      operator-(const Quantity<OtherNumber, Unit>& other) const {
    return Quantity<decltype(Number() - OtherNumber()), Unit>(value - other.value, unsafe);
  }
  template <typename OtherNumber, typename = EnableIf<isIntegralOrBounded<OtherNumber>()>>
  inline constexpr Quantity<decltype(Number() * OtherNumber()), Unit>
      operator*(OtherNumber other) const {
    return Quantity<decltype(Number() * other), Unit>(value * other, unsafe);
  }
  template <typename OtherNumber, typename = EnableIf<isIntegralOrBounded<OtherNumber>()>>
  inline constexpr Quantity<decltype(Number() / OtherNumber()), Unit>
      operator/(OtherNumber other) const {
    return Quantity<decltype(Number() / other), Unit>(value / other, unsafe);
  }
  template <typename OtherNumber>
  inline constexpr decltype(Number() / OtherNumber())
      operator/(const Quantity<OtherNumber, Unit>& other) const {
    return value / other.value;
  }
  template <typename OtherNumber>
  inline constexpr Quantity<decltype(Number() % OtherNumber()), Unit>
      operator%(const Quantity<OtherNumber, Unit>& other) const {
    return Quantity<decltype(Number() % OtherNumber()), Unit>(value % other.value, unsafe);
  }

  template <typename OtherNumber, typename OtherUnit>
  inline constexpr Quantity<decltype(Number() * OtherNumber()), OtherUnit>
      operator*(UnitRatio<OtherNumber, OtherUnit, Unit> ratio) const {
    return Quantity<decltype(Number() * OtherNumber()), OtherUnit>(
        value * ratio.unit1PerUnit2, unsafe);
  }
  template <typename OtherNumber, typename OtherUnit>
  inline constexpr Quantity<decltype(Number() / OtherNumber()), OtherUnit>
      operator/(UnitRatio<OtherNumber, Unit, OtherUnit> ratio) const {
    return Quantity<decltype(Number() / OtherNumber()), OtherUnit>(
        value / ratio.unit1PerUnit2, unsafe);
  }
  template <typename OtherNumber, typename OtherUnit>
  inline constexpr Quantity<decltype(Number() % OtherNumber()), Unit>
      operator%(UnitRatio<OtherNumber, Unit, OtherUnit> ratio) const {
    return Quantity<decltype(Number() % OtherNumber()), Unit>(
        value % ratio.unit1PerUnit2, unsafe);
  }
  template <typename OtherNumber, typename OtherUnit>
  inline constexpr UnitRatio<decltype(Number() / OtherNumber()), Unit, OtherUnit>
      operator/(Quantity<OtherNumber, OtherUnit> other) const {
    return UnitRatio<decltype(Number() / OtherNumber()), Unit, OtherUnit>(
        value / other.value, unsafe);
  }

  template <typename OtherNumber>
  inline constexpr bool operator==(const Quantity<OtherNumber, Unit>& other) const {
    return value == other.value;
  }
  template <typename OtherNumber>
  inline constexpr bool operator!=(const Quantity<OtherNumber, Unit>& other) const {
    return value != other.value;
  }
  template <typename OtherNumber>
  inline constexpr bool operator<=(const Quantity<OtherNumber, Unit>& other) const {
    return value <= other.value;
  }
  template <typename OtherNumber>
  inline constexpr bool operator>=(const Quantity<OtherNumber, Unit>& other) const {
    return value >= other.value;
  }
  template <typename OtherNumber>
  inline constexpr bool operator<(const Quantity<OtherNumber, Unit>& other) const {
    return value < other.value;
  }
  template <typename OtherNumber>
  inline constexpr bool operator>(const Quantity<OtherNumber, Unit>& other) const {
    return value > other.value;
  }

  template <typename OtherNumber>
  inline Quantity& operator+=(const Quantity<OtherNumber, Unit>& other) {
    value += other.value;
    return *this;
  }
  template <typename OtherNumber>
  inline Quantity& operator-=(const Quantity<OtherNumber, Unit>& other) {
    value -= other.value;
    return *this;
  }
  template <typename OtherNumber>
  inline Quantity& operator*=(OtherNumber other) {
    value *= other;
    return *this;
  }
  template <typename OtherNumber>
  inline Quantity& operator/=(OtherNumber other) {
    value /= other.value;
    return *this;
  }

private:
  Number value;

  template <typename OtherNumber, typename OtherUnit>
  friend class Quantity;

  template <typename Number1, typename Number2, typename Unit2, typename>
  friend inline constexpr auto operator*(Number1 a, Quantity<Number2, Unit2> b)
      -> Quantity<decltype(Number1() * Number2()), Unit2>;
};

template <typename T> struct Unit_ {
  static inline constexpr T get() { return T(1); }
};
template <typename T, typename U>
struct Unit_<Quantity<T, U>> {
  static inline constexpr Quantity<decltype(Unit_<T>::get()), U> get() {
    return Quantity<decltype(Unit_<T>::get()), U>(Unit_<T>::get(), unsafe);
  }
};

template <typename T>
inline constexpr auto unit() -> decltype(Unit_<T>::get()) { return Unit_<T>::get(); }
// unit<Quantity<T, U>>() returns a Quantity of value 1.  It also, intentionally, works on basic
// numeric types.

template <typename Number1, typename Number2, typename Unit,
          typename = EnableIf<isIntegralOrBounded<Number1>()>>
inline constexpr auto operator*(Number1 a, Quantity<Number2, Unit> b)
    -> Quantity<decltype(Number1() * Number2()), Unit> {
  return Quantity<decltype(Number1() * Number2()), Unit>(a * b.value, unsafe);
}

template <typename Number1, typename Number2, typename Unit, typename Unit2>
inline constexpr auto operator*(UnitRatio<Number1, Unit2, Unit> ratio,
    Quantity<Number2, Unit> measure)
    -> decltype(measure * ratio) {
  return measure * ratio;
}

// =======================================================================================
// Absolute measures

template <typename T, typename Label>
class Absolute {
  // Wraps some other value -- typically a Quantity -- but represents a value measured based on
  // some absolute origin.  For example, if `Duration` is a type representing a time duration,
  // Absolute<Duration, UnixEpoch> might be a calendar date.
  //
  // Since Absolute represents measurements relative to some arbitrary origin, the only sensible
  // arithmetic to perform on them is addition and subtraction.

  // TODO(someday):  Do the same automatic expansion of integer width that Quantity does?  Doesn't
  //   matter for our time use case, where we always use 64-bit anyway.  Note that fixing this
  //   would implicitly allow things like multiplying an Absolute by a UnitRatio to change its
  //   units, which is actually totally logical and kind of neat.

public:
  inline constexpr Absolute(MaxValue_): value(maxValue) {}
  inline constexpr Absolute(MinValue_): value(minValue) {}
  // Allow initialization from maxValue and minValue.
  // TODO(msvc): decltype(maxValue) and decltype(minValue) deduce unknown-type for these function
  // parameters, causing the compiler to complain of a duplicate constructor definition, so we
  // specify MaxValue_ and MinValue_ types explicitly.

  inline constexpr Absolute operator+(const T& other) const { return Absolute(value + other); }
  inline constexpr Absolute operator-(const T& other) const { return Absolute(value - other); }
  inline constexpr T operator-(const Absolute& other) const { return value - other.value; }

  inline Absolute& operator+=(const T& other) { value += other; return *this; }
  inline Absolute& operator-=(const T& other) { value -= other; return *this; }

  inline constexpr bool operator==(const Absolute& other) const { return value == other.value; }
  inline constexpr bool operator!=(const Absolute& other) const { return value != other.value; }
  inline constexpr bool operator<=(const Absolute& other) const { return value <= other.value; }
  inline constexpr bool operator>=(const Absolute& other) const { return value >= other.value; }
  inline constexpr bool operator< (const Absolute& other) const { return value <  other.value; }
  inline constexpr bool operator> (const Absolute& other) const { return value >  other.value; }

private:
  T value;

  explicit constexpr Absolute(T value): value(value) {}

  template <typename U>
  friend inline constexpr U origin();
};

template <typename T, typename Label>
inline constexpr Absolute<T, Label> operator+(const T& a, const Absolute<T, Label>& b) {
  return b + a;
}

template <typename T> struct UnitOf_ { typedef T Type; };
template <typename T, typename Label> struct UnitOf_<Absolute<T, Label>> { typedef T Type; };
template <typename T>
using UnitOf = typename UnitOf_<T>::Type;
// UnitOf<Absolute<T, U>> is T.  UnitOf<AnythingElse> is AnythingElse.

template <typename T>
inline constexpr T origin() { return T(0 * unit<UnitOf<T>>()); }
// origin<Absolute<T, U>>() returns an Absolute of value 0.  It also, intentionally, works on basic
// numeric types.

// =======================================================================================
// Overflow avoidance

template <uint64_t n, uint accum = 0>
struct BitCount_ {
  static constexpr uint value = BitCount_<(n >> 1), accum + 1>::value;
};
template <uint accum>
struct BitCount_<0, accum> {
  static constexpr uint value = accum;
};

template <uint64_t n>
inline constexpr uint bitCount() { return BitCount_<n>::value; }
// Number of bits required to represent the number `n`.

template <uint bitCountBitCount> struct AtLeastUInt_ {
  static_assert(bitCountBitCount < 7, "don't know how to represent integers over 64 bits");
};
template <> struct AtLeastUInt_<0> { typedef uint8_t Type; };
template <> struct AtLeastUInt_<1> { typedef uint8_t Type; };
template <> struct AtLeastUInt_<2> { typedef uint8_t Type; };
template <> struct AtLeastUInt_<3> { typedef uint8_t Type; };
template <> struct AtLeastUInt_<4> { typedef uint16_t Type; };
template <> struct AtLeastUInt_<5> { typedef uint32_t Type; };
template <> struct AtLeastUInt_<6> { typedef uint64_t Type; };

template <uint bits>
using AtLeastUInt = typename AtLeastUInt_<bitCount<max(bits, 1) - 1>()>::Type;
// AtLeastUInt<n> is an unsigned integer of at least n bits. E.g. AtLeastUInt<12> is uint16_t.

// -------------------------------------------------------------------

template <uint value>
class BoundedConst {
  // A constant integer value on which we can do bit size analysis.

public:
  BoundedConst() = default;

  inline constexpr uint unwrap() const { return value; }

#define OP(op, check) \
  template <uint other> \
  inline constexpr BoundedConst<(value op other)> \
      operator op(BoundedConst<other>) const { \
    static_assert(check, "overflow in BoundedConst arithmetic"); \
    return BoundedConst<(value op other)>(); \
  }
#define COMPARE_OP(op) \
  template <uint other> \
  inline constexpr bool operator op(BoundedConst<other>) const { \
    return value op other; \
  }

  OP(+, value + other >= value)
  OP(-, value - other <= value)
  OP(*, value * other / other == value)
  OP(/, true)   // div by zero already errors out; no other division ever overflows
  OP(%, true)   // mod by zero already errors out; no other modulus ever overflows
  OP(<<, value << other >= value)
  OP(>>, true)  // right shift can't overflow
  OP(&, true)   // bitwise ops can't overflow
  OP(|, true)   // bitwise ops can't overflow

  COMPARE_OP(==)
  COMPARE_OP(!=)
  COMPARE_OP(< )
  COMPARE_OP(> )
  COMPARE_OP(<=)
  COMPARE_OP(>=)
#undef OP
#undef COMPARE_OP
};

template <uint64_t m, typename T>
struct Unit_<Bounded<m, T>> {
  static inline constexpr BoundedConst<1> get() { return BoundedConst<1>(); }
};

template <uint value>
struct Unit_<BoundedConst<value>> {
  static inline constexpr BoundedConst<1> get() { return BoundedConst<1>(); }
};

template <uint value>
inline constexpr BoundedConst<value> bounded() {
  return BoundedConst<value>();
}

template <uint64_t a, uint64_t b>
static constexpr uint64_t boundedAdd() {
  static_assert(a + b >= a, "possible overflow detected");
  return a + b;
}
template <uint64_t a, uint64_t b>
static constexpr uint64_t boundedSub() {
  static_assert(a - b <= a, "possible underflow detected");
  return a - b;
}
template <uint64_t a, uint64_t b>
static constexpr uint64_t boundedMul() {
  static_assert(a * b / b == a, "possible overflow detected");
  return a * b;
}
template <uint64_t a, uint64_t b>
static constexpr uint64_t boundedLShift() {
  static_assert(a << b >= a, "possible overflow detected");
  return a << b;
}

template <uint a, uint b>
inline constexpr BoundedConst<kj::min(a, b)> min(BoundedConst<a>, BoundedConst<b>) {
  return bounded<kj::min(a, b)>();
}
template <uint a, uint b>
inline constexpr BoundedConst<kj::max(a, b)> max(BoundedConst<a>, BoundedConst<b>) {
  return bounded<kj::max(a, b)>();
}
// We need to override min() and max() between constants because the ternary operator in the
// default implementation would complain.

// -------------------------------------------------------------------

template <uint64_t maxN, typename T>
class Bounded {
public:
  static_assert(maxN <= T(kj::maxValue), "possible overflow detected");

  Bounded() = default;

  Bounded(const Bounded& other) = default;
  template <typename OtherInt, typename = EnableIf<isIntegral<OtherInt>()>>
  inline constexpr Bounded(OtherInt value): value(value) {
    static_assert(OtherInt(maxValue) <= maxN, "possible overflow detected");
  }
  template <uint64_t otherMax, typename OtherT>
  inline constexpr Bounded(const Bounded<otherMax, OtherT>& other)
      : value(other.value) {
    static_assert(otherMax <= maxN, "possible overflow detected");
  }
  template <uint otherValue>
  inline constexpr Bounded(BoundedConst<otherValue>)
      : value(otherValue) {
    static_assert(otherValue <= maxN, "overflow detected");
  }

  Bounded& operator=(const Bounded& other) = default;
  template <typename OtherInt, typename = EnableIf<isIntegral<OtherInt>()>>
  Bounded& operator=(OtherInt other) {
    static_assert(OtherInt(maxValue) <= maxN, "possible overflow detected");
    value = other;
    return *this;
  }
  template <uint64_t otherMax, typename OtherT>
  inline Bounded& operator=(const Bounded<otherMax, OtherT>& other) {
    static_assert(otherMax <= maxN, "possible overflow detected");
    value = other.value;
    return *this;
  }
  template <uint otherValue>
  inline Bounded& operator=(BoundedConst<otherValue>) {
    static_assert(otherValue <= maxN, "overflow detected");
    value = otherValue;
    return *this;
  }

  inline constexpr T unwrap() const { return value; }

#define OP(op, newMax) \
  template <uint64_t otherMax, typename otherT> \
  inline constexpr Bounded<newMax, decltype(T() op otherT())> \
      operator op(const Bounded<otherMax, otherT>& other) const { \
    return Bounded<newMax, decltype(T() op otherT())>(value op other.value, unsafe); \
  }
#define COMPARE_OP(op) \
  template <uint64_t otherMax, typename OtherT> \
  inline constexpr bool operator op(const Bounded<otherMax, OtherT>& other) const { \
    return value op other.value; \
  }

  OP(+, (boundedAdd<maxN, otherMax>()))
  OP(*, (boundedMul<maxN, otherMax>()))
  OP(/, maxN)
  OP(%, otherMax - 1)

  // operator- is intentionally omitted because we mostly use this with unsigned types, and
  // subtraction requires proof that subtrahend is not greater than the minuend.

  COMPARE_OP(==)
  COMPARE_OP(!=)
  COMPARE_OP(< )
  COMPARE_OP(> )
  COMPARE_OP(<=)
  COMPARE_OP(>=)

#undef OP
#undef COMPARE_OP

  template <uint64_t newMax, typename ErrorFunc>
  inline Bounded<newMax, T> assertMax(ErrorFunc&& func) const {
    // Assert that the number is no more than `newMax`. Otherwise, call `func`.
    static_assert(newMax < maxN, "this bounded size assertion is redundant");
    if (KJ_UNLIKELY(value > newMax)) func();
    return Bounded<newMax, T>(value, unsafe);
  }

  template <uint64_t otherMax, typename OtherT, typename ErrorFunc>
  inline Bounded<maxN, decltype(T() - OtherT())> subtractChecked(
      const Bounded<otherMax, OtherT>& other, ErrorFunc&& func) const {
    // Subtract a number, calling func() if the result would underflow.
    if (KJ_UNLIKELY(value < other.value)) func();
    return Bounded<maxN, decltype(T() - OtherT())>(value - other.value, unsafe);
  }

  template <uint otherValue, typename ErrorFunc>
  inline Bounded<maxN - otherValue, T> subtractChecked(
      BoundedConst<otherValue>, ErrorFunc&& func) const {
    // Subtract a number, calling func() if the result would underflow.
    static_assert(otherValue <= maxN, "underflow detected");
    if (KJ_UNLIKELY(value < otherValue)) func();
    return Bounded<maxN - otherValue, T>(value - otherValue, unsafe);
  }

  template <uint64_t otherMax, typename OtherT>
  inline Maybe<Bounded<maxN, decltype(T() - OtherT())>> trySubtract(
      const Bounded<otherMax, OtherT>& other) const {
    // Subtract a number, calling func() if the result would underflow.
    if (value < other.value) {
      return nullptr;
    } else {
      return Bounded<maxN, decltype(T() - OtherT())>(value - other.value, unsafe);
    }
  }

  template <uint otherValue>
  inline Maybe<Bounded<maxN - otherValue, T>> trySubtract(BoundedConst<otherValue>) const {
    // Subtract a number, calling func() if the result would underflow.
    if (value < otherValue) {
      return nullptr;
    } else {
      return Bounded<maxN - otherValue, T>(value - otherValue, unsafe);
    }
  }

  inline constexpr Bounded(T value, decltype(unsafe)): value(value) {}
  template <uint64_t otherMax, typename OtherT>
  inline constexpr Bounded(Bounded<otherMax, OtherT> value, decltype(unsafe))
      : value(value.value) {}
  // Mainly for internal use.
  //
  // Only use these as a last resort, with ample commentary on why you think it's safe.

private:
  T value;

  template <uint64_t, typename>
  friend class Bounded;
};

template <typename Number>
inline constexpr Bounded<Number(kj::maxValue), Number> bounded(Number value) {
  return Bounded<Number(kj::maxValue), Number>(value, unsafe);
}

inline constexpr Bounded<1, uint8_t> bounded(bool value) {
  return Bounded<1, uint8_t>(value, unsafe);
}

template <uint bits, typename Number>
inline constexpr Bounded<maxValueForBits<bits>(), Number> assumeBits(Number value) {
  return Bounded<maxValueForBits<bits>(), Number>(value, unsafe);
}

template <uint bits, uint64_t maxN, typename T>
inline constexpr Bounded<maxValueForBits<bits>(), T> assumeBits(Bounded<maxN, T> value) {
  return Bounded<maxValueForBits<bits>(), T>(value, unsafe);
}

template <uint bits, typename Number, typename Unit>
inline constexpr auto assumeBits(Quantity<Number, Unit> value)
    -> Quantity<decltype(assumeBits<bits>(value / unit<Quantity<Number, Unit>>())), Unit> {
  return Quantity<decltype(assumeBits<bits>(value / unit<Quantity<Number, Unit>>())), Unit>(
      assumeBits<bits>(value / unit<Quantity<Number, Unit>>()), unsafe);
}

template <uint64_t maxN, typename Number>
inline constexpr Bounded<maxN, Number> assumeMax(Number value) {
  return Bounded<maxN, Number>(value, unsafe);
}

template <uint64_t newMaxN, uint64_t maxN, typename T>
inline constexpr Bounded<newMaxN, T> assumeMax(Bounded<maxN, T> value) {
  return Bounded<newMaxN, T>(value, unsafe);
}

template <uint64_t maxN, typename Number, typename Unit>
inline constexpr auto assumeMax(Quantity<Number, Unit> value)
    -> Quantity<decltype(assumeMax<maxN>(value / unit<Quantity<Number, Unit>>())), Unit> {
  return Quantity<decltype(assumeMax<maxN>(value / unit<Quantity<Number, Unit>>())), Unit>(
      assumeMax<maxN>(value / unit<Quantity<Number, Unit>>()), unsafe);
}

template <uint maxN, typename Number>
inline constexpr Bounded<maxN, Number> assumeMax(BoundedConst<maxN>, Number value) {
  return assumeMax<maxN>(value);
}

template <uint newMaxN, uint64_t maxN, typename T>
inline constexpr Bounded<newMaxN, T> assumeMax(BoundedConst<maxN>, Bounded<maxN, T> value) {
  return assumeMax<maxN>(value);
}

template <uint maxN, typename Number, typename Unit>
inline constexpr auto assumeMax(Quantity<BoundedConst<maxN>, Unit>, Quantity<Number, Unit> value)
    -> decltype(assumeMax<maxN>(value)) {
  return assumeMax<maxN>(value);
}

template <uint64_t newMax, uint64_t maxN, typename T, typename ErrorFunc>
inline Bounded<newMax, T> assertMax(Bounded<maxN, T> value, ErrorFunc&& errorFunc) {
  // Assert that the bounded value is less than or equal to the given maximum, calling errorFunc()
  // if not.
  static_assert(newMax < maxN, "this bounded size assertion is redundant");
  return value.template assertMax<newMax>(kj::fwd<ErrorFunc>(errorFunc));
}

template <uint64_t newMax, uint64_t maxN, typename T, typename Unit, typename ErrorFunc>
inline Quantity<Bounded<newMax, T>, Unit> assertMax(
    Quantity<Bounded<maxN, T>, Unit> value, ErrorFunc&& errorFunc) {
  // Assert that the bounded value is less than or equal to the given maximum, calling errorFunc()
  // if not.
  static_assert(newMax < maxN, "this bounded size assertion is redundant");
  return (value / unit<decltype(value)>()).template assertMax<newMax>(
      kj::fwd<ErrorFunc>(errorFunc)) * unit<decltype(value)>();
}

template <uint newMax, uint64_t maxN, typename T, typename ErrorFunc>
inline Bounded<newMax, T> assertMax(
    BoundedConst<newMax>, Bounded<maxN, T> value, ErrorFunc&& errorFunc) {
  return assertMax<newMax>(value, kj::mv(errorFunc));
}

template <uint newMax, uint64_t maxN, typename T, typename Unit, typename ErrorFunc>
inline Quantity<Bounded<newMax, T>, Unit> assertMax(
    Quantity<BoundedConst<newMax>, Unit>,
    Quantity<Bounded<maxN, T>, Unit> value, ErrorFunc&& errorFunc) {
  return assertMax<newMax>(value, kj::mv(errorFunc));
}

template <uint64_t newBits, uint64_t maxN, typename T, typename ErrorFunc = ThrowOverflow>
inline Bounded<maxValueForBits<newBits>(), T> assertMaxBits(
    Bounded<maxN, T> value, ErrorFunc&& errorFunc = ErrorFunc()) {
  // Assert that the bounded value requires no more than the given number of bits, calling
  // errorFunc() if not.
  return assertMax<maxValueForBits<newBits>()>(value, kj::fwd<ErrorFunc>(errorFunc));
}

template <uint64_t newBits, uint64_t maxN, typename T, typename Unit,
          typename ErrorFunc = ThrowOverflow>
inline Quantity<Bounded<maxValueForBits<newBits>(), T>, Unit> assertMaxBits(
    Quantity<Bounded<maxN, T>, Unit> value, ErrorFunc&& errorFunc = ErrorFunc()) {
  // Assert that the bounded value requires no more than the given number of bits, calling
  // errorFunc() if not.
  return assertMax<maxValueForBits<newBits>()>(value, kj::fwd<ErrorFunc>(errorFunc));
}

template <typename newT, uint64_t maxN, typename T>
inline constexpr Bounded<maxN, newT> upgradeBound(Bounded<maxN, T> value) {
  return value;
}

template <typename newT, uint64_t maxN, typename T, typename Unit>
inline constexpr Quantity<Bounded<maxN, newT>, Unit> upgradeBound(
    Quantity<Bounded<maxN, T>, Unit> value) {
  return value;
}

template <uint64_t maxN, typename T, typename Other, typename ErrorFunc>
inline auto subtractChecked(Bounded<maxN, T> value, Other other, ErrorFunc&& errorFunc)
    -> decltype(value.subtractChecked(other, kj::fwd<ErrorFunc>(errorFunc))) {
  return value.subtractChecked(other, kj::fwd<ErrorFunc>(errorFunc));
}

template <typename T, typename U, typename Unit, typename ErrorFunc>
inline auto subtractChecked(Quantity<T, Unit> value, Quantity<U, Unit> other, ErrorFunc&& errorFunc)
    -> Quantity<decltype(subtractChecked(T(), U(), kj::fwd<ErrorFunc>(errorFunc))), Unit> {
  return subtractChecked(value / unit<Quantity<T, Unit>>(),
                         other / unit<Quantity<U, Unit>>(),
                         kj::fwd<ErrorFunc>(errorFunc))
      * unit<Quantity<T, Unit>>();
}

template <uint64_t maxN, typename T, typename Other>
inline auto trySubtract(Bounded<maxN, T> value, Other other)
    -> decltype(value.trySubtract(other)) {
  return value.trySubtract(other);
}

template <typename T, typename U, typename Unit>
inline auto trySubtract(Quantity<T, Unit> value, Quantity<U, Unit> other)
    -> Maybe<Quantity<decltype(subtractChecked(T(), U(), int())), Unit>> {
  return trySubtract(value / unit<Quantity<T, Unit>>(),
                     other / unit<Quantity<U, Unit>>())
      .map([](decltype(subtractChecked(T(), U(), int())) x) {
    return x * unit<Quantity<T, Unit>>();
  });
}

template <uint64_t aN, uint64_t bN, typename A, typename B>
inline constexpr Bounded<kj::min(aN, bN), WiderType<A, B>>
min(Bounded<aN, A> a, Bounded<bN, B> b) {
  return Bounded<kj::min(aN, bN), WiderType<A, B>>(kj::min(a.unwrap(), b.unwrap()), unsafe);
}
template <uint64_t aN, uint64_t bN, typename A, typename B>
inline constexpr Bounded<kj::max(aN, bN), WiderType<A, B>>
max(Bounded<aN, A> a, Bounded<bN, B> b) {
  return Bounded<kj::max(aN, bN), WiderType<A, B>>(kj::max(a.unwrap(), b.unwrap()), unsafe);
}
// We need to override min() and max() because:
// 1) WiderType<> might not choose the correct bounds.
// 2) One of the two sides of the ternary operator in the default implementation would fail to
//    typecheck even though it is OK in practice.

// -------------------------------------------------------------------
// Operators between Bounded and BoundedConst

#define OP(op, newMax) \
template <uint64_t maxN, uint cvalue, typename T> \
inline constexpr Bounded<(newMax), decltype(T() op uint())> operator op( \
    Bounded<maxN, T> value, BoundedConst<cvalue>) { \
  return Bounded<(newMax), decltype(T() op uint())>(value.unwrap() op cvalue, unsafe); \
}

#define REVERSE_OP(op, newMax) \
template <uint64_t maxN, uint cvalue, typename T> \
inline constexpr Bounded<(newMax), decltype(uint() op T())> operator op( \
    BoundedConst<cvalue>, Bounded<maxN, T> value) { \
  return Bounded<(newMax), decltype(uint() op T())>(cvalue op value.unwrap(), unsafe); \
}

#define COMPARE_OP(op) \
template <uint64_t maxN, uint cvalue, typename T> \
inline constexpr bool operator op(Bounded<maxN, T> value, BoundedConst<cvalue>) { \
  return value.unwrap() op cvalue; \
} \
template <uint64_t maxN, uint cvalue, typename T> \
inline constexpr bool operator op(BoundedConst<cvalue>, Bounded<maxN, T> value) { \
  return cvalue op value.unwrap(); \
}

OP(+, (boundedAdd<maxN, cvalue>()))
REVERSE_OP(+, (boundedAdd<maxN, cvalue>()))

OP(*, (boundedMul<maxN, cvalue>()))
REVERSE_OP(*, (boundedAdd<maxN, cvalue>()))

OP(/, maxN / cvalue)
REVERSE_OP(/, cvalue)  // denominator could be 1

OP(%, cvalue - 1)
REVERSE_OP(%, maxN - 1)

OP(<<, (boundedLShift<maxN, cvalue>()))
REVERSE_OP(<<, (boundedLShift<cvalue, maxN>()))

OP(>>, maxN >> cvalue)
REVERSE_OP(>>, cvalue >> maxN)

OP(&, maxValueForBits<bitCount<maxN>()>() & cvalue)
REVERSE_OP(&, maxValueForBits<bitCount<maxN>()>() & cvalue)

OP(|, maxN | cvalue)
REVERSE_OP(|, maxN | cvalue)

COMPARE_OP(==)
COMPARE_OP(!=)
COMPARE_OP(< )
COMPARE_OP(> )
COMPARE_OP(<=)
COMPARE_OP(>=)

#undef OP
#undef REVERSE_OP
#undef COMPARE_OP

template <uint64_t maxN, uint cvalue, typename T>
inline constexpr Bounded<cvalue, decltype(uint() - T())>
    operator-(BoundedConst<cvalue>, Bounded<maxN, T> value) {
  // We allow subtraction of a variable from a constant only if the constant is greater than or
  // equal to the maximum possible value of the variable. Since the variable could be zero, the
  // result can be as large as the constant.
  //
  // We do not allow subtraction of a constant from a variable because there's never a guarantee it
  // won't underflow (unless the constant is zero, which is silly).
  static_assert(cvalue >= maxN, "possible underflow detected");
  return Bounded<cvalue, decltype(uint() - T())>(cvalue - value.unwrap(), unsafe);
}

template <uint64_t aN, uint b, typename A>
inline constexpr Bounded<kj::min(aN, b), A> min(Bounded<aN, A> a, BoundedConst<b>) {
  return Bounded<kj::min(aN, b), A>(kj::min(b, a.unwrap()), unsafe);
}
template <uint64_t aN, uint b, typename A>
inline constexpr Bounded<kj::min(aN, b), A> min(BoundedConst<b>, Bounded<aN, A> a) {
  return Bounded<kj::min(aN, b), A>(kj::min(a.unwrap(), b), unsafe);
}
template <uint64_t aN, uint b, typename A>
inline constexpr Bounded<kj::max(aN, b), A> max(Bounded<aN, A> a, BoundedConst<b>) {
  return Bounded<kj::max(aN, b), A>(kj::max(b, a.unwrap()), unsafe);
}
template <uint64_t aN, uint b, typename A>
inline constexpr Bounded<kj::max(aN, b), A> max(BoundedConst<b>, Bounded<aN, A> a) {
  return Bounded<kj::max(aN, b), A>(kj::max(a.unwrap(), b), unsafe);
}
// We need to override min() between a Bounded and a constant since:
// 1) WiderType<> might choose BoundedConst over a 1-byte Bounded, which is wrong.
// 2) To clamp the bounds of the output type.
// 3) Same ternary operator typechecking issues.

// -------------------------------------------------------------------

template <uint64_t maxN, typename T>
class SafeUnwrapper {
public:
  inline explicit constexpr SafeUnwrapper(Bounded<maxN, T> value): value(value.unwrap()) {}

  template <typename U, typename = EnableIf<isIntegral<U>()>>
  inline constexpr operator U() const {
    static_assert(maxN <= U(maxValue), "possible truncation detected");
    return value;
  }

  inline constexpr operator bool() const {
    static_assert(maxN <= 1, "possible truncation detected");
    return value;
  }

private:
  T value;
};

template <uint64_t maxN, typename T>
inline constexpr SafeUnwrapper<maxN, T> unbound(Bounded<maxN, T> bounded) {
  // Unwraps the bounded value, returning a value that can be implicitly cast to any integer type.
  // If this implicit cast could truncate, a compile-time error will be raised.
  return SafeUnwrapper<maxN, T>(bounded);
}

template <uint64_t value>
class SafeConstUnwrapper {
public:
  template <typename T, typename = EnableIf<isIntegral<T>()>>
  inline constexpr operator T() const {
    static_assert(value <= T(maxValue), "this operation will truncate");
    return value;
  }

  inline constexpr operator bool() const {
    static_assert(value <= 1, "this operation will truncate");
    return value;
  }
};

template <uint value>
inline constexpr SafeConstUnwrapper<value> unbound(BoundedConst<value>) {
  return SafeConstUnwrapper<value>();
}

template <typename T, typename U>
inline constexpr T unboundAs(U value) {
  return unbound(value);
}

template <uint64_t requestedMax, uint64_t maxN, typename T>
inline constexpr T unboundMax(Bounded<maxN, T> value) {
  // Explicitly unguard expecting a value that is at most `maxN`.
  static_assert(maxN <= requestedMax, "possible overflow detected");
  return value.unwrap();
}

template <uint64_t requestedMax, uint value>
inline constexpr uint unboundMax(BoundedConst<value>) {
  // Explicitly unguard expecting a value that is at most `maxN`.
  static_assert(value <= requestedMax, "overflow detected");
  return value;
}

template <uint bits, typename T>
inline constexpr auto unboundMaxBits(T value) ->
    decltype(unboundMax<maxValueForBits<bits>()>(value)) {
  // Explicitly unguard expecting a value that fits into `bits` bits.
  return unboundMax<maxValueForBits<bits>()>(value);
}

#define OP(op) \
template <uint64_t maxN, typename T, typename U> \
inline constexpr auto operator op(T a, SafeUnwrapper<maxN, U> b) -> decltype(a op (T)b) { \
  return a op (AtLeastUInt<sizeof(T)*8>)b; \
} \
template <uint64_t maxN, typename T, typename U> \
inline constexpr auto operator op(SafeUnwrapper<maxN, U> b, T a) -> decltype((T)b op a) { \
  return (AtLeastUInt<sizeof(T)*8>)b op a; \
} \
template <uint64_t value, typename T> \
inline constexpr auto operator op(T a, SafeConstUnwrapper<value> b) -> decltype(a op (T)b) { \
  return a op (AtLeastUInt<sizeof(T)*8>)b; \
} \
template <uint64_t value, typename T> \
inline constexpr auto operator op(SafeConstUnwrapper<value> b, T a) -> decltype((T)b op a) { \
  return (AtLeastUInt<sizeof(T)*8>)b op a; \
}

OP(+)
OP(-)
OP(*)
OP(/)
OP(%)
OP(<<)
OP(>>)
OP(&)
OP(|)
OP(==)
OP(!=)
OP(<=)
OP(>=)
OP(<)
OP(>)

#undef OP

// -------------------------------------------------------------------

template <uint64_t maxN, typename T>
class Range<Bounded<maxN, T>> {
public:
  inline constexpr Range(Bounded<maxN, T> begin, Bounded<maxN, T> end)
      : inner(unbound(begin), unbound(end)) {}
  inline explicit constexpr Range(Bounded<maxN, T> end)
      : inner(unbound(end)) {}

  class Iterator {
  public:
    Iterator() = default;
    inline explicit Iterator(typename Range<T>::Iterator inner): inner(inner) {}

    inline Bounded<maxN, T> operator* () const { return Bounded<maxN, T>(*inner, unsafe); }
    inline Iterator& operator++() { ++inner; return *this; }

    inline bool operator==(const Iterator& other) const { return inner == other.inner; }
    inline bool operator!=(const Iterator& other) const { return inner != other.inner; }

  private:
    typename Range<T>::Iterator inner;
  };

  inline Iterator begin() const { return Iterator(inner.begin()); }
  inline Iterator end() const { return Iterator(inner.end()); }

private:
  Range<T> inner;
};

template <typename T, typename U>
class Range<Quantity<T, U>> {
public:
  inline constexpr Range(Quantity<T, U> begin, Quantity<T, U> end)
      : inner(begin / unit<Quantity<T, U>>(), end / unit<Quantity<T, U>>()) {}
  inline explicit constexpr Range(Quantity<T, U> end)
      : inner(end / unit<Quantity<T, U>>()) {}

  class Iterator {
  public:
    Iterator() = default;
    inline explicit Iterator(typename Range<T>::Iterator inner): inner(inner) {}

    inline Quantity<T, U> operator* () const { return *inner * unit<Quantity<T, U>>(); }
    inline Iterator& operator++() { ++inner; return *this; }

    inline bool operator==(const Iterator& other) const { return inner == other.inner; }
    inline bool operator!=(const Iterator& other) const { return inner != other.inner; }

  private:
    typename Range<T>::Iterator inner;
  };

  inline Iterator begin() const { return Iterator(inner.begin()); }
  inline Iterator end() const { return Iterator(inner.end()); }

private:
  Range<T> inner;
};

template <uint value>
inline constexpr Range<Bounded<value, uint>> zeroTo(BoundedConst<value> end) {
  return Range<Bounded<value, uint>>(end);
}

template <uint value, typename Unit>
inline constexpr Range<Quantity<Bounded<value, uint>, Unit>>
    zeroTo(Quantity<BoundedConst<value>, Unit> end) {
  return Range<Quantity<Bounded<value, uint>, Unit>>(end);
}

}  // namespace kj

KJ_END_HEADER
