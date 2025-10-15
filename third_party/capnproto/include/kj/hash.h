// Copyright (c) 2018 Kenton Varda and contributors
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

#include "string.h"

KJ_BEGIN_HEADER

namespace kj {
namespace _ {  // private

struct HashCoder {
  // This is a dummy type with only one instance: HASHCODER (below).  To make an arbitrary type
  // hashable, define `operator*(HashCoder, T)` to return any other type that is already hashable.
  // Be sure to declare the operator in the same namespace as `T` **or** in the global scope.
  // You can use the KJ_HASHCODE() macro as syntax sugar for this.
  //
  // A more usual way to accomplish what we're doing here would be to require that you define
  // a function like `hashCode(T)` and then rely on argument-dependent lookup.  However, this has
  // the problem that it pollutes other people's namespaces and even the global namespace.  For
  // example, some other project may already have functions called `hashCode` which do something
  // different.  Declaring `operator*` with `HashCoder` as the left operand cannot conflict with
  // anything.

  uint operator*(ArrayPtr<const byte> s) const;
  inline uint operator*(ArrayPtr<byte> s) const { return operator*(s.asConst()); }

  inline uint operator*(ArrayPtr<const char> s) const { return operator*(s.asBytes()); }
  inline uint operator*(ArrayPtr<char> s) const { return operator*(s.asBytes()); }
  inline uint operator*(const Array<const char>& s) const { return operator*(s.asBytes()); }
  inline uint operator*(const Array<char>& s) const { return operator*(s.asBytes()); }
  inline uint operator*(const String& s) const { return operator*(s.asBytes()); }
  inline uint operator*(const StringPtr& s) const { return operator*(s.asBytes()); }
  inline uint operator*(const ConstString& s) const { return operator*(s.asBytes()); }

  inline uint operator*(decltype(nullptr)) const { return 0; }
  inline uint operator*(bool b) const { return b; }
  inline uint operator*(char i) const { return i; }
  inline uint operator*(signed char i) const { return i; }
  inline uint operator*(unsigned char i) const { return i; }
  inline uint operator*(signed short i) const { return i; }
  inline uint operator*(unsigned short i) const { return i; }
  inline uint operator*(signed int i) const { return i; }
  inline uint operator*(unsigned int i) const { return i; }

  inline uint operator*(signed long i) const {
    if (sizeof(i) == sizeof(uint)) {
      return operator*(static_cast<uint>(i));
    } else {
      return operator*(static_cast<unsigned long long>(i));
    }
  }
  inline uint operator*(unsigned long i) const {
    if (sizeof(i) == sizeof(uint)) {
      return operator*(static_cast<uint>(i));
    } else {
      return operator*(static_cast<unsigned long long>(i));
    }
  }
  inline uint operator*(signed long long i) const {
    return operator*(static_cast<unsigned long long>(i));
  }
  inline uint operator*(unsigned long long i) const {
    // Mix 64 bits to 32 bits in such a way that if our input values differ primarily in the upper
    // 32 bits, we still get good diffusion. (I.e. we cannot just truncate!)
    //
    // 49123 is an arbitrarily-chosen prime that is vaguely close to 2^16.
    //
    // TODO(perf): I just made this up. Is it OK?
    return static_cast<uint>(i) + static_cast<uint>(i >> 32) * 49123;
  }

  template <typename T>
  uint operator*(T* ptr) const {
    static_assert(!isSameType<Decay<T>, char>(), "Wrap in StringPtr if you want to hash string "
        "contents. If you want to hash the pointer, cast to void*");
    if (sizeof(ptr) == sizeof(uint)) {
      // TODO(cleanup): In C++17, make the if() above be `if constexpr ()`, then change this to
      //   reinterpret_cast<uint>(ptr).
      return reinterpret_cast<unsigned long long>(ptr);
    } else {
      return operator*(reinterpret_cast<unsigned long long>(ptr));
    }
  }

  template <typename T, typename = decltype(instance<const HashCoder&>() * instance<const T&>())>
  uint operator*(ArrayPtr<T> arr) const;
  template <typename T, typename = decltype(instance<const HashCoder&>() * instance<const T&>())>
  uint operator*(const Array<T>& arr) const;
  template <typename T, typename = EnableIf<__is_enum(T)>>
  inline uint operator*(T e) const;

  template <typename T, typename Result = decltype(instance<T>().hashCode())>
  inline Result operator*(T&& value) const { return kj::fwd<T>(value).hashCode(); }
};
static KJ_CONSTEXPR(const) HashCoder HASHCODER = HashCoder();

}  // namespace _ (private)

#define KJ_HASHCODE(...) operator*(::kj::_::HashCoder, __VA_ARGS__)
// Defines a hash function for a custom type.  Example:
//
//    class Foo {...};
//    inline uint KJ_HASHCODE(const Foo& foo) { return kj::hashCode(foo.x, foo.y); }
//
// This allows Foo to be passed to hashCode().
//
// The function should be declared either in the same namespace as the target type or in the global
// namespace. It can return any type which itself is hashable -- that value will be hashed in turn
// until a `uint` comes out.

inline uint hashCode(uint value) { return value; }
template <typename T>
inline uint hashCode(T&& value) { return hashCode(_::HASHCODER * kj::fwd<T>(value)); }
template <typename T, size_t N>
inline uint hashCode(T (&arr)[N]) {
  static_assert(!isSameType<Decay<T>, char>(), "Wrap in StringPtr if you want to hash string "
      "contents. If you want to hash the pointer, cast to void*");
  static_assert(isSameType<Decay<T>, char>(), "Wrap in ArrayPtr if you want to hash a C array. "
      "If you want to hash the pointer, cast to void*");
  return 0;
}
template <typename... T>
inline uint hashCode(T&&... values) {
  uint hashes[] = { hashCode(kj::fwd<T>(values))... };
  return hashCode(kj::ArrayPtr<uint>(hashes).asBytes());
}
// kj::hashCode() is a universal hashing function, like kj::str() is a universal stringification
// function. Throw stuff in, get a hash code.
//
// Hash codes may differ between different processes, even running exactly the same code.
//
// NOT SUITABLE FOR CRYPTOGRAPHY. This is for hash tables, not crypto.

// =======================================================================================
// inline implementation details

namespace _ {  // private

template <typename T, typename>
inline uint HashCoder::operator*(ArrayPtr<T> arr) const {
  // Hash each array element to create a string of hashes, then murmur2 over those.
  //
  // TODO(perf): Choose a more-modern hash. (See hash.c++.)

  constexpr uint m = 0x5bd1e995;
  constexpr uint r = 24;
  uint h = arr.size() * sizeof(uint);

  for (auto& e: arr) {
    uint k = kj::hashCode(e);
    k *= m;
    k ^= k >> r;
    k *= m;
    h *= m;
    h ^= k;
  }

  h ^= h >> 13;
  h *= m;
  h ^= h >> 15;
  return h;
}
template <typename T, typename>
inline uint HashCoder::operator*(const Array<T>& arr) const {
  return operator*(arr.asPtr());
}

template <typename T, typename>
inline uint HashCoder::operator*(T e) const {
  return operator*(static_cast<__underlying_type(T)>(e));
}

}  // namespace _ (private)
} // namespace kj

KJ_END_HEADER
