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

#include "array.h"

KJ_BEGIN_HEADER

namespace kj {

template <typename T>
class Vector {
  // Similar to std::vector, but based on KJ framework.
  //
  // This implementation always uses move constructors when growing the backing array.  If the
  // move constructor throws, the Vector is left in an inconsistent state.  This is acceptable
  // under KJ exception theory which assumes that exceptions leave things in inconsistent states.

  // TODO(someday): Allow specifying a custom allocator.

public:
  inline Vector() = default;
  inline explicit Vector(size_t capacity): builder(heapArrayBuilder<T>(capacity)) {}
  inline Vector(Array<T>&& array): builder(kj::mv(array)) {}

  inline operator ArrayPtr<T>() KJ_LIFETIMEBOUND { return builder; }
  inline operator ArrayPtr<const T>() const KJ_LIFETIMEBOUND { return builder; }
  inline ArrayPtr<T> asPtr() KJ_LIFETIMEBOUND { return builder.asPtr(); }
  inline ArrayPtr<const T> asPtr() const KJ_LIFETIMEBOUND { return builder.asPtr(); }

  inline size_t size() const { return builder.size(); }
  inline bool empty() const { return size() == 0; }
  inline size_t capacity() const { return builder.capacity(); }
  inline T& operator[](size_t index) KJ_LIFETIMEBOUND { return builder[index]; }
  inline const T& operator[](size_t index) const KJ_LIFETIMEBOUND { return builder[index]; }

  inline const T* begin() const KJ_LIFETIMEBOUND { return builder.begin(); }
  inline const T* end() const KJ_LIFETIMEBOUND { return builder.end(); }
  inline const T& front() const KJ_LIFETIMEBOUND { return builder.front(); }
  inline const T& back() const KJ_LIFETIMEBOUND { return builder.back(); }
  inline T* begin() KJ_LIFETIMEBOUND { return builder.begin(); }
  inline T* end() KJ_LIFETIMEBOUND { return builder.end(); }
  inline T& front() KJ_LIFETIMEBOUND { return builder.front(); }
  inline T& back() KJ_LIFETIMEBOUND { return builder.back(); }

  inline Array<T> releaseAsArray() {
    // TODO(perf):  Avoid a copy/move by allowing Array<T> to point to incomplete space?
    if (!builder.isFull()) {
      setCapacity(size());
    }
    return builder.finish();
  }

  template <typename U>
  inline bool operator==(const U& other) const { return asPtr() == other; }
  template <typename U>
  inline bool operator!=(const U& other) const { return asPtr() != other; }

  inline ArrayPtr<T> slice(size_t start, size_t end) KJ_LIFETIMEBOUND {
    return asPtr().slice(start, end);
  }
  inline ArrayPtr<const T> slice(size_t start, size_t end) const KJ_LIFETIMEBOUND {
    return asPtr().slice(start, end);
  }

  template <typename... Params>
  inline T& add(Params&&... params) KJ_LIFETIMEBOUND {
    if (builder.isFull()) grow();
    return builder.add(kj::fwd<Params>(params)...);
  }

  template <typename Iterator>
  inline void addAll(Iterator begin, Iterator end) {
    size_t needed = builder.size() + (end - begin);
    if (needed > builder.capacity()) grow(needed);
    builder.addAll(begin, end);
  }

  template <typename Container>
  inline void addAll(Container&& container) {
    addAll(container.begin(), container.end());
  }

  inline void removeLast() {
    builder.removeLast();
  }

  inline void resize(size_t size) {
    if (size > builder.capacity()) grow(size);
    builder.resize(size);
  }

  inline void operator=(decltype(nullptr)) {
    builder = nullptr;
  }

  inline void clear() {
    builder.clear();
  }

  inline void truncate(size_t size) {
    builder.truncate(size);
  }

  inline void reserve(size_t size) {
    if (size > builder.capacity()) {
      grow(size);
    }
  }

private:
  ArrayBuilder<T> builder;

  void grow(size_t minCapacity = 0) {
    setCapacity(kj::max(minCapacity, capacity() == 0 ? 4 : capacity() * 2));
  }
  void setCapacity(size_t newSize) {
    if (builder.size() > newSize) {
      builder.truncate(newSize);
    }
    ArrayBuilder<T> newBuilder = heapArrayBuilder<T>(newSize);
    newBuilder.addAll(kj::mv(builder));
    builder = kj::mv(newBuilder);
  }
};

template <typename T>
inline auto KJ_STRINGIFY(const Vector<T>& v) -> decltype(toCharSequence(v.asPtr())) {
  return toCharSequence(v.asPtr());
}

}  // namespace kj

KJ_END_HEADER
