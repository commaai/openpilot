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

#ifndef KJ_ONE_OF_H_
#define KJ_ONE_OF_H_

#if defined(__GNUC__) && !KJ_HEADER_WARNINGS
#pragma GCC system_header
#endif

#include "common.h"

namespace kj {

namespace _ {  // private

template <uint i, typename Key, typename First, typename... Rest>
struct TypeIndex_ { static constexpr uint value = TypeIndex_<i + 1, Key, Rest...>::value; };
template <uint i, typename Key, typename... Rest>
struct TypeIndex_<i, Key, Key, Rest...> { static constexpr uint value = i; };

}  // namespace _ (private)

template <typename... Variants>
class OneOf {
  template <typename Key>
  static inline constexpr uint typeIndex() { return _::TypeIndex_<1, Key, Variants...>::value; }
  // Get the 1-based index of Key within the type list Types.

public:
  inline OneOf(): tag(0) {}
  OneOf(const OneOf& other) { copyFrom(other); }
  OneOf(OneOf&& other) { moveFrom(other); }
  ~OneOf() { destroy(); }

  OneOf& operator=(const OneOf& other) { if (tag != 0) destroy(); copyFrom(other); return *this; }
  OneOf& operator=(OneOf&& other) { if (tag != 0) destroy(); moveFrom(other); return *this; }

  inline bool operator==(decltype(nullptr)) const { return tag == 0; }
  inline bool operator!=(decltype(nullptr)) const { return tag != 0; }

  template <typename T>
  bool is() const {
    return tag == typeIndex<T>();
  }

  template <typename T>
  T& get() {
    KJ_IREQUIRE(is<T>(), "Must check OneOf::is<T>() before calling get<T>().");
    return *reinterpret_cast<T*>(space);
  }
  template <typename T>
  const T& get() const {
    KJ_IREQUIRE(is<T>(), "Must check OneOf::is<T>() before calling get<T>().");
    return *reinterpret_cast<const T*>(space);
  }

  template <typename T, typename... Params>
  T& init(Params&&... params) {
    if (tag != 0) destroy();
    ctor(*reinterpret_cast<T*>(space), kj::fwd<Params>(params)...);
    tag = typeIndex<T>();
    return *reinterpret_cast<T*>(space);
  }

private:
  uint tag;

  static inline constexpr size_t maxSize(size_t a) {
    return a;
  }
  template <typename... Rest>
  static inline constexpr size_t maxSize(size_t a, size_t b, Rest... rest) {
    return maxSize(kj::max(a, b), rest...);
  }
  // Returns the maximum of all the parameters.
  // TODO(someday):  Generalize the above template and make it common.  I tried, but C++ decided to
  //   be difficult so I cut my losses.

  static constexpr auto spaceSize = maxSize(sizeof(Variants)...);
  // TODO(msvc):  This constant could just as well go directly inside space's bracket's, where it's
  // used, but MSVC suffers a parse error on `...`.

  union {
    byte space[spaceSize];

    void* forceAligned;
    // TODO(someday):  Use C++11 alignas() once we require GCC 4.8 / Clang 3.3.
  };

  template <typename... T>
  inline void doAll(T... t) {}

  template <typename T>
  inline bool destroyVariant() {
    if (tag == typeIndex<T>()) {
      tag = 0;
      dtor(*reinterpret_cast<T*>(space));
    }
    return false;
  }
  void destroy() {
    doAll(destroyVariant<Variants>()...);
  }

  template <typename T>
  inline bool copyVariantFrom(const OneOf& other) {
    if (other.is<T>()) {
      ctor(*reinterpret_cast<T*>(space), other.get<T>());
    }
    return false;
  }
  void copyFrom(const OneOf& other) {
    // Initialize as a copy of `other`.  Expects that `this` starts out uninitialized, so the tag
    // is invalid.
    tag = other.tag;
    doAll(copyVariantFrom<Variants>(other)...);
  }

  template <typename T>
  inline bool moveVariantFrom(OneOf& other) {
    if (other.is<T>()) {
      ctor(*reinterpret_cast<T*>(space), kj::mv(other.get<T>()));
    }
    return false;
  }
  void moveFrom(OneOf& other) {
    // Initialize as a copy of `other`.  Expects that `this` starts out uninitialized, so the tag
    // is invalid.
    tag = other.tag;
    doAll(moveVariantFrom<Variants>(other)...);
  }
};

}  // namespace kj

#endif  // KJ_ONE_OF_H_
