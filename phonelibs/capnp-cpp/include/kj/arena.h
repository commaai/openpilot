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

#ifndef KJ_ARENA_H_
#define KJ_ARENA_H_

#if defined(__GNUC__) && !KJ_HEADER_WARNINGS
#pragma GCC system_header
#endif

#include "memory.h"
#include "array.h"
#include "string.h"

namespace kj {

class Arena {
  // A class which allows several objects to be allocated in contiguous chunks of memory, then
  // frees them all at once.
  //
  // Allocating from the same Arena in multiple threads concurrently is NOT safe, because making
  // it safe would require atomic operations that would slow down allocation even when
  // single-threaded.  If you need to use arena allocation in a multithreaded context, consider
  // allocating thread-local arenas.

public:
  explicit Arena(size_t chunkSizeHint = 1024);
  // Create an Arena.  `chunkSizeHint` hints at where to start when allocating chunks, but is only
  // a hint -- the Arena will, for example, allocate progressively larger chunks as time goes on,
  // in order to reduce overall allocation overhead.

  explicit Arena(ArrayPtr<byte> scratch);
  // Allocates from the given scratch space first, only resorting to the heap when it runs out.

  KJ_DISALLOW_COPY(Arena);
  ~Arena() noexcept(false);

  template <typename T, typename... Params>
  T& allocate(Params&&... params);
  template <typename T>
  ArrayPtr<T> allocateArray(size_t size);
  // Allocate an object or array of type T.  If T has a non-trivial destructor, that destructor
  // will be run during the Arena's destructor.  Such destructors are run in opposite order of
  // allocation.  Note that these methods must maintain a list of destructors to call, which has
  // overhead, but this overhead only applies if T has a non-trivial destructor.

  template <typename T, typename... Params>
  Own<T> allocateOwn(Params&&... params);
  template <typename T>
  Array<T> allocateOwnArray(size_t size);
  template <typename T>
  ArrayBuilder<T> allocateOwnArrayBuilder(size_t capacity);
  // Allocate an object or array of type T.  Destructors are executed when the returned Own<T>
  // or Array<T> goes out-of-scope, which must happen before the Arena is destroyed.  This variant
  // is useful when you need to control when the destructor is called.  This variant also avoids
  // the need for the Arena itself to keep track of destructors to call later, which may make it
  // slightly more efficient.

  template <typename T>
  inline T& copy(T&& value) { return allocate<Decay<T>>(kj::fwd<T>(value)); }
  // Allocate a copy of the given value in the arena.  This is just a shortcut for calling the
  // type's copy (or move) constructor.

  StringPtr copyString(StringPtr content);
  // Make a copy of the given string inside the arena, and return a pointer to the copy.

private:
  struct ChunkHeader {
    ChunkHeader* next;
    byte* pos;  // first unallocated byte in this chunk
    byte* end;  // end of this chunk
  };
  struct ObjectHeader {
    void (*destructor)(void*);
    ObjectHeader* next;
  };

  size_t nextChunkSize;
  ChunkHeader* chunkList = nullptr;
  ObjectHeader* objectList = nullptr;

  ChunkHeader* currentChunk = nullptr;

  void cleanup();
  // Run all destructors, leaving the above pointers null.  If a destructor throws, the State is
  // left in a consistent state, such that if cleanup() is called again, it will pick up where
  // it left off.

  void* allocateBytes(size_t amount, uint alignment, bool hasDisposer);
  // Allocate the given number of bytes.  `hasDisposer` must be true if `setDisposer()` may be
  // called on this pointer later.

  void* allocateBytesInternal(size_t amount, uint alignment);
  // Try to allocate the given number of bytes without taking a lock.  Fails if and only if there
  // is no space left in the current chunk.

  void setDestructor(void* ptr, void (*destructor)(void*));
  // Schedule the given destructor to be executed when the Arena is destroyed.  `ptr` must be a
  // pointer previously returned by an `allocateBytes()` call for which `hasDisposer` was true.

  template <typename T>
  static void destroyArray(void* pointer) {
    size_t elementCount = *reinterpret_cast<size_t*>(pointer);
    constexpr size_t prefixSize = kj::max(alignof(T), sizeof(size_t));
    DestructorOnlyArrayDisposer::instance.disposeImpl(
        reinterpret_cast<byte*>(pointer) + prefixSize,
        sizeof(T), elementCount, elementCount, &destroyObject<T>);
  }

  template <typename T>
  static void destroyObject(void* pointer) {
    dtor(*reinterpret_cast<T*>(pointer));
  }
};

// =======================================================================================
// Inline implementation details

template <typename T, typename... Params>
T& Arena::allocate(Params&&... params) {
  T& result = *reinterpret_cast<T*>(allocateBytes(
      sizeof(T), alignof(T), !__has_trivial_destructor(T)));
  if (!__has_trivial_constructor(T) || sizeof...(Params) > 0) {
    ctor(result, kj::fwd<Params>(params)...);
  }
  if (!__has_trivial_destructor(T)) {
    setDestructor(&result, &destroyObject<T>);
  }
  return result;
}

template <typename T>
ArrayPtr<T> Arena::allocateArray(size_t size) {
  if (__has_trivial_destructor(T)) {
    ArrayPtr<T> result =
        arrayPtr(reinterpret_cast<T*>(allocateBytes(
            sizeof(T) * size, alignof(T), false)), size);
    if (!__has_trivial_constructor(T)) {
      for (size_t i = 0; i < size; i++) {
        ctor(result[i]);
      }
    }
    return result;
  } else {
    // Allocate with a 64-bit prefix in which we store the array size.
    constexpr size_t prefixSize = kj::max(alignof(T), sizeof(size_t));
    void* base = allocateBytes(sizeof(T) * size + prefixSize, alignof(T), true);
    size_t& tag = *reinterpret_cast<size_t*>(base);
    ArrayPtr<T> result =
        arrayPtr(reinterpret_cast<T*>(reinterpret_cast<byte*>(base) + prefixSize), size);
    setDestructor(base, &destroyArray<T>);

    if (__has_trivial_constructor(T)) {
      tag = size;
    } else {
      // In case of constructor exceptions, we need the tag to end up storing the number of objects
      // that were successfully constructed, so that they'll be properly destroyed.
      tag = 0;
      for (size_t i = 0; i < size; i++) {
        ctor(result[i]);
        tag = i + 1;
      }
    }
    return result;
  }
}

template <typename T, typename... Params>
Own<T> Arena::allocateOwn(Params&&... params) {
  T& result = *reinterpret_cast<T*>(allocateBytes(sizeof(T), alignof(T), false));
  if (!__has_trivial_constructor(T) || sizeof...(Params) > 0) {
    ctor(result, kj::fwd<Params>(params)...);
  }
  return Own<T>(&result, DestructorOnlyDisposer<T>::instance);
}

template <typename T>
Array<T> Arena::allocateOwnArray(size_t size) {
  ArrayBuilder<T> result = allocateOwnArrayBuilder<T>(size);
  for (size_t i = 0; i < size; i++) {
    result.add();
  }
  return result.finish();
}

template <typename T>
ArrayBuilder<T> Arena::allocateOwnArrayBuilder(size_t capacity) {
  return ArrayBuilder<T>(
      reinterpret_cast<T*>(allocateBytes(sizeof(T) * capacity, alignof(T), false)),
      capacity, DestructorOnlyArrayDisposer::instance);
}

}  // namespace kj

#endif  // KJ_ARENA_H_
