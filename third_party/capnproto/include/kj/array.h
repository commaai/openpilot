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
#include <string.h>
#include <initializer_list>

KJ_BEGIN_HEADER

namespace kj {

// =======================================================================================
// ArrayDisposer -- Implementation details.

class ArrayDisposer {
  // Much like Disposer from memory.h.

protected:
  // Do not declare a destructor, as doing so will force a global initializer for
  // HeapArrayDisposer::instance.

  virtual void disposeImpl(void* firstElement, size_t elementSize, size_t elementCount,
                           size_t capacity, void (*destroyElement)(void*)) const = 0;
  // Disposes of the array.  `destroyElement` invokes the destructor of each element, or is nullptr
  // if the elements have trivial destructors.  `capacity` is the amount of space that was
  // allocated while `elementCount` is the number of elements that were actually constructed;
  // these are always the same number for Array<T> but may be different when using ArrayBuilder<T>.

public:

  template <typename T>
  void dispose(T* firstElement, size_t elementCount, size_t capacity) const;
  // Helper wrapper around disposeImpl().
  //
  // Callers must not call dispose() on the same array twice, even if the first call throws
  // an exception.

private:
  template <typename T, bool hasTrivialDestructor = KJ_HAS_TRIVIAL_DESTRUCTOR(T)>
  struct Dispose_;
};

class ExceptionSafeArrayUtil {
  // Utility class that assists in constructing or destroying elements of an array, where the
  // constructor or destructor could throw exceptions.  In case of an exception,
  // ExceptionSafeArrayUtil's destructor will call destructors on all elements that have been
  // constructed but not destroyed.  Remember that destructors that throw exceptions are required
  // to use UnwindDetector to detect unwind and avoid exceptions in this case.  Therefore, no more
  // than one exception will be thrown (and the program will not terminate).

public:
  inline ExceptionSafeArrayUtil(void* ptr, size_t elementSize, size_t constructedElementCount,
                                void (*destroyElement)(void*))
      : pos(reinterpret_cast<byte*>(ptr) + elementSize * constructedElementCount),
        elementSize(elementSize), constructedElementCount(constructedElementCount),
        destroyElement(destroyElement) {}
  KJ_DISALLOW_COPY_AND_MOVE(ExceptionSafeArrayUtil);

  inline ~ExceptionSafeArrayUtil() noexcept(false) {
    if (constructedElementCount > 0) destroyAll();
  }

  void construct(size_t count, void (*constructElement)(void*));
  // Construct the given number of elements.

  void destroyAll();
  // Destroy all elements.  Call this immediately before ExceptionSafeArrayUtil goes out-of-scope
  // to ensure that one element throwing an exception does not prevent the others from being
  // destroyed.

  void release() { constructedElementCount = 0; }
  // Prevent ExceptionSafeArrayUtil's destructor from destroying the constructed elements.
  // Call this after you've successfully finished constructing.

private:
  byte* pos;
  size_t elementSize;
  size_t constructedElementCount;
  void (*destroyElement)(void*);
};

class DestructorOnlyArrayDisposer: public ArrayDisposer {
public:
  static const DestructorOnlyArrayDisposer instance;

  void disposeImpl(void* firstElement, size_t elementSize, size_t elementCount,
                   size_t capacity, void (*destroyElement)(void*)) const override;
};

class NullArrayDisposer: public ArrayDisposer {
  // An ArrayDisposer that does nothing.  Can be used to construct a fake Arrays that doesn't
  // actually own its content.

public:
  static const NullArrayDisposer instance;

  void disposeImpl(void* firstElement, size_t elementSize, size_t elementCount,
                   size_t capacity, void (*destroyElement)(void*)) const override;
};

// =======================================================================================
// Array

template <typename T>
class Array {
  // An owned array which will automatically be disposed of (using an ArrayDisposer) in the
  // destructor.  Can be moved, but not copied.  Much like Own<T>, but for arrays rather than
  // single objects.

public:
  inline Array(): ptr(nullptr), size_(0), disposer(nullptr) {}
  inline Array(decltype(nullptr)): ptr(nullptr), size_(0), disposer(nullptr) {}
  inline Array(Array&& other) noexcept
      : ptr(other.ptr), size_(other.size_), disposer(other.disposer) {
    other.ptr = nullptr;
    other.size_ = 0;
  }
  inline Array(Array<RemoveConstOrDisable<T>>&& other) noexcept
      : ptr(other.ptr), size_(other.size_), disposer(other.disposer) {
    other.ptr = nullptr;
    other.size_ = 0;
  }
  inline Array(T* firstElement KJ_LIFETIMEBOUND, size_t size, const ArrayDisposer& disposer)
      : ptr(firstElement), size_(size), disposer(&disposer) {}

  KJ_DISALLOW_COPY(Array);
  inline ~Array() noexcept { dispose(); }

  inline operator ArrayPtr<T>() KJ_LIFETIMEBOUND {
    return ArrayPtr<T>(ptr, size_);
  }
  inline operator ArrayPtr<const T>() const KJ_LIFETIMEBOUND {
    return ArrayPtr<T>(ptr, size_);
  }
  inline ArrayPtr<T> asPtr() KJ_LIFETIMEBOUND {
    return ArrayPtr<T>(ptr, size_);
  }
  inline ArrayPtr<const T> asPtr() const KJ_LIFETIMEBOUND {
    return ArrayPtr<T>(ptr, size_);
  }

  inline size_t size() const { return size_; }
  inline T& operator[](size_t index) KJ_LIFETIMEBOUND {
    KJ_IREQUIRE(index < size_, "Out-of-bounds Array access.");
    return ptr[index];
  }
  inline const T& operator[](size_t index) const KJ_LIFETIMEBOUND {
    KJ_IREQUIRE(index < size_, "Out-of-bounds Array access.");
    return ptr[index];
  }

  inline const T* begin() const KJ_LIFETIMEBOUND { return ptr; }
  inline const T* end() const KJ_LIFETIMEBOUND { return ptr + size_; }
  inline const T& front() const KJ_LIFETIMEBOUND { return *ptr; }
  inline const T& back() const KJ_LIFETIMEBOUND { return *(ptr + size_ - 1); }
  inline T* begin() KJ_LIFETIMEBOUND { return ptr; }
  inline T* end() KJ_LIFETIMEBOUND { return ptr + size_; }
  inline T& front() KJ_LIFETIMEBOUND { return *ptr; }
  inline T& back() KJ_LIFETIMEBOUND { return *(ptr + size_ - 1); }

  template <typename U>
  inline bool operator==(const U& other) const { return asPtr() == other; }
  template <typename U>
  inline bool operator!=(const U& other) const { return asPtr() != other; }

  inline ArrayPtr<T> slice(size_t start, size_t end) KJ_LIFETIMEBOUND {
    KJ_IREQUIRE(start <= end && end <= size_, "Out-of-bounds Array::slice().");
    return ArrayPtr<T>(ptr + start, end - start);
  }
  inline ArrayPtr<const T> slice(size_t start, size_t end) const KJ_LIFETIMEBOUND {
    KJ_IREQUIRE(start <= end && end <= size_, "Out-of-bounds Array::slice().");
    return ArrayPtr<const T>(ptr + start, end - start);
  }

  inline ArrayPtr<const byte> asBytes() const KJ_LIFETIMEBOUND { return asPtr().asBytes(); }
  inline ArrayPtr<PropagateConst<T, byte>> asBytes() KJ_LIFETIMEBOUND { return asPtr().asBytes(); }
  inline ArrayPtr<const char> asChars() const KJ_LIFETIMEBOUND { return asPtr().asChars(); }
  inline ArrayPtr<PropagateConst<T, char>> asChars() KJ_LIFETIMEBOUND { return asPtr().asChars(); }

  inline Array<PropagateConst<T, byte>> releaseAsBytes() {
    // Like asBytes() but transfers ownership.
    static_assert(sizeof(T) == sizeof(byte),
        "releaseAsBytes() only possible on arrays with byte-size elements (e.g. chars).");
    Array<PropagateConst<T, byte>> result(
        reinterpret_cast<PropagateConst<T, byte>*>(ptr), size_, *disposer);
    ptr = nullptr;
    size_ = 0;
    return result;
  }
  inline Array<PropagateConst<T, char>> releaseAsChars() {
    // Like asChars() but transfers ownership.
    static_assert(sizeof(T) == sizeof(PropagateConst<T, char>),
        "releaseAsChars() only possible on arrays with char-size elements (e.g. bytes).");
    Array<PropagateConst<T, char>> result(
        reinterpret_cast<PropagateConst<T, char>*>(ptr), size_, *disposer);
    ptr = nullptr;
    size_ = 0;
    return result;
  }

  inline bool operator==(decltype(nullptr)) const { return size_ == 0; }
  inline bool operator!=(decltype(nullptr)) const { return size_ != 0; }

  inline Array& operator=(decltype(nullptr)) {
    dispose();
    return *this;
  }

  inline Array& operator=(Array&& other) {
    dispose();
    ptr = other.ptr;
    size_ = other.size_;
    disposer = other.disposer;
    other.ptr = nullptr;
    other.size_ = 0;
    return *this;
  }

  template <typename... Attachments>
  Array<T> attach(Attachments&&... attachments) KJ_WARN_UNUSED_RESULT;
  // Like Own<T>::attach(), but attaches to an Array.

private:
  T* ptr;
  size_t size_;
  const ArrayDisposer* disposer;

  inline void dispose() {
    // Make sure that if an exception is thrown, we are left with a null ptr, so we won't possibly
    // dispose again.
    T* ptrCopy = ptr;
    size_t sizeCopy = size_;
    if (ptrCopy != nullptr) {
      ptr = nullptr;
      size_ = 0;
      disposer->dispose(ptrCopy, sizeCopy, sizeCopy);
    }
  }

  template <typename U>
  friend class Array;
  template <typename U>
  friend class ArrayBuilder;
};

static_assert(!canMemcpy<Array<char>>(), "canMemcpy<>() is broken");

namespace _ {  // private

class HeapArrayDisposer final: public ArrayDisposer {
public:
  template <typename T>
  static T* allocate(size_t count);
  template <typename T>
  static T* allocateUninitialized(size_t count);

  static const HeapArrayDisposer instance;

private:
  static void* allocateImpl(size_t elementSize, size_t elementCount, size_t capacity,
                            void (*constructElement)(void*), void (*destroyElement)(void*));
  // Allocates and constructs the array.  Both function pointers are null if the constructor is
  // trivial, otherwise destroyElement is null if the constructor doesn't throw.

  virtual void disposeImpl(void* firstElement, size_t elementSize, size_t elementCount,
                           size_t capacity, void (*destroyElement)(void*)) const override;

  template <typename T, bool hasTrivialConstructor = KJ_HAS_TRIVIAL_CONSTRUCTOR(T),
                        bool hasNothrowConstructor = KJ_HAS_NOTHROW_CONSTRUCTOR(T)>
  struct Allocate_;
};

}  // namespace _ (private)

template <typename T>
inline Array<T> heapArray(size_t size) {
  // Much like `heap<T>()` from memory.h, allocates a new array on the heap.

  return Array<T>(_::HeapArrayDisposer::allocate<T>(size), size,
                  _::HeapArrayDisposer::instance);
}

template <typename T> Array<T> heapArray(const T* content, size_t size);
template <typename T> Array<T> heapArray(ArrayPtr<T> content);
template <typename T> Array<T> heapArray(ArrayPtr<const T> content);
template <typename T, typename Iterator> Array<T> heapArray(Iterator begin, Iterator end);
template <typename T> Array<T> heapArray(std::initializer_list<T> init);
// Allocate a heap array containing a copy of the given content.

template <typename T, typename Container>
Array<T> heapArrayFromIterable(Container&& a) { return heapArray<T>(a.begin(), a.end()); }
template <typename T>
Array<T> heapArrayFromIterable(Array<T>&& a) { return mv(a); }

// =======================================================================================
// ArrayBuilder

template <typename T>
class ArrayBuilder {
  // Class which lets you build an Array<T> specifying the exact constructor arguments for each
  // element, rather than starting by default-constructing them.

public:
  ArrayBuilder(): ptr(nullptr), pos(nullptr), endPtr(nullptr) {}
  ArrayBuilder(decltype(nullptr)): ptr(nullptr), pos(nullptr), endPtr(nullptr) {}
  explicit ArrayBuilder(RemoveConst<T>* firstElement, size_t capacity,
                        const ArrayDisposer& disposer)
      : ptr(firstElement), pos(firstElement), endPtr(firstElement + capacity),
        disposer(&disposer) {}
  ArrayBuilder(ArrayBuilder&& other)
      : ptr(other.ptr), pos(other.pos), endPtr(other.endPtr), disposer(other.disposer) {
    other.ptr = nullptr;
    other.pos = nullptr;
    other.endPtr = nullptr;
  }
  ArrayBuilder(Array<T>&& other)
      : ptr(other.ptr), pos(other.ptr + other.size_), endPtr(pos), disposer(other.disposer) {
    // Create an already-full ArrayBuilder from an Array of the same type. This constructor
    // primarily exists to enable Vector<T> to be constructed from Array<T>.
    other.ptr = nullptr;
    other.size_ = 0;
  }
  KJ_DISALLOW_COPY(ArrayBuilder);
  inline ~ArrayBuilder() noexcept(false) { dispose(); }

  inline operator ArrayPtr<T>() KJ_LIFETIMEBOUND {
    return arrayPtr(ptr, pos);
  }
  inline operator ArrayPtr<const T>() const KJ_LIFETIMEBOUND {
    return arrayPtr(ptr, pos);
  }
  inline ArrayPtr<T> asPtr() KJ_LIFETIMEBOUND {
    return arrayPtr(ptr, pos);
  }
  inline ArrayPtr<const T> asPtr() const KJ_LIFETIMEBOUND {
    return arrayPtr(ptr, pos);
  }

  inline size_t size() const { return pos - ptr; }
  inline size_t capacity() const { return endPtr - ptr; }
  inline T& operator[](size_t index) KJ_LIFETIMEBOUND {
    KJ_IREQUIRE(index < implicitCast<size_t>(pos - ptr), "Out-of-bounds Array access.");
    return ptr[index];
  }
  inline const T& operator[](size_t index) const KJ_LIFETIMEBOUND {
    KJ_IREQUIRE(index < implicitCast<size_t>(pos - ptr), "Out-of-bounds Array access.");
    return ptr[index];
  }

  inline const T* begin() const KJ_LIFETIMEBOUND { return ptr; }
  inline const T* end() const KJ_LIFETIMEBOUND { return pos; }
  inline const T& front() const KJ_LIFETIMEBOUND { return *ptr; }
  inline const T& back() const KJ_LIFETIMEBOUND { return *(pos - 1); }
  inline T* begin() KJ_LIFETIMEBOUND { return ptr; }
  inline T* end() KJ_LIFETIMEBOUND { return pos; }
  inline T& front() KJ_LIFETIMEBOUND { return *ptr; }
  inline T& back() KJ_LIFETIMEBOUND { return *(pos - 1); }

  ArrayBuilder& operator=(ArrayBuilder&& other) {
    dispose();
    ptr = other.ptr;
    pos = other.pos;
    endPtr = other.endPtr;
    disposer = other.disposer;
    other.ptr = nullptr;
    other.pos = nullptr;
    other.endPtr = nullptr;
    return *this;
  }
  ArrayBuilder& operator=(decltype(nullptr)) {
    dispose();
    return *this;
  }

  template <typename... Params>
  T& add(Params&&... params) KJ_LIFETIMEBOUND {
    KJ_IREQUIRE(pos < endPtr, "Added too many elements to ArrayBuilder.");
    ctor(*pos, kj::fwd<Params>(params)...);
    return *pos++;
  }

  template <typename Container>
  void addAll(Container&& container) {
    addAll<decltype(container.begin()), !isReference<Container>()>(
        container.begin(), container.end());
  }

  template <typename Iterator, bool move = false>
  void addAll(Iterator start, Iterator end);

  void removeLast() {
    KJ_IREQUIRE(pos > ptr, "No elements present to remove.");
    kj::dtor(*--pos);
  }

  void truncate(size_t size) {
    KJ_IREQUIRE(size <= this->size(), "can't use truncate() to expand");

    T* target = ptr + size;
    if (KJ_HAS_TRIVIAL_DESTRUCTOR(T)) {
      pos = target;
    } else {
      while (pos > target) {
        kj::dtor(*--pos);
      }
    }
  }

  void clear() {
    if (KJ_HAS_TRIVIAL_DESTRUCTOR(T)) {
      pos = ptr;
    } else {
      while (pos > ptr) {
        kj::dtor(*--pos);
      }
    }
  }

  void resize(size_t size) {
    KJ_IREQUIRE(size <= capacity(), "can't resize past capacity");

    T* target = ptr + size;
    if (target > pos) {
      // expand
      if (KJ_HAS_TRIVIAL_CONSTRUCTOR(T)) {
        pos = target;
      } else {
        while (pos < target) {
          kj::ctor(*pos++);
        }
      }
    } else {
      // truncate
      if (KJ_HAS_TRIVIAL_DESTRUCTOR(T)) {
        pos = target;
      } else {
        while (pos > target) {
          kj::dtor(*--pos);
        }
      }
    }
  }

  Array<T> finish() {
    // We could safely remove this check if we assume that the disposer implementation doesn't
    // need to know the original capacity, as is the case with HeapArrayDisposer since it uses
    // operator new() or if we created a custom disposer for ArrayBuilder which stores the capacity
    // in a prefix.  But that would make it hard to write cleverer heap allocators, and anyway this
    // check might catch bugs.  Probably people should use Vector if they want to build arrays
    // without knowing the final size in advance.
    KJ_IREQUIRE(pos == endPtr, "ArrayBuilder::finish() called prematurely.");
    Array<T> result(reinterpret_cast<T*>(ptr), pos - ptr, *disposer);
    ptr = nullptr;
    pos = nullptr;
    endPtr = nullptr;
    return result;
  }

  inline bool isFull() const {
    return pos == endPtr;
  }

private:
  T* ptr;
  RemoveConst<T>* pos;
  T* endPtr;
  const ArrayDisposer* disposer = &NullArrayDisposer::instance;

  inline void dispose() {
    // Make sure that if an exception is thrown, we are left with a null ptr, so we won't possibly
    // dispose again.
    T* ptrCopy = ptr;
    T* posCopy = pos;
    T* endCopy = endPtr;
    if (ptrCopy != nullptr) {
      ptr = nullptr;
      pos = nullptr;
      endPtr = nullptr;
      disposer->dispose(ptrCopy, posCopy - ptrCopy, endCopy - ptrCopy);
    }
  }
};

template <typename T>
inline ArrayBuilder<T> heapArrayBuilder(size_t size) {
  // Like `heapArray<T>()` but does not default-construct the elements.  You must construct them
  // manually by calling `add()`.

  return ArrayBuilder<T>(_::HeapArrayDisposer::allocateUninitialized<RemoveConst<T>>(size),
                         size, _::HeapArrayDisposer::instance);
}

// =======================================================================================
// Inline Arrays

template <typename T, size_t fixedSize>
class FixedArray {
  // A fixed-width array whose storage is allocated inline rather than on the heap.

public:
  inline constexpr size_t size() const { return fixedSize; }
  inline constexpr T* begin() KJ_LIFETIMEBOUND { return content; }
  inline constexpr T* end() KJ_LIFETIMEBOUND { return content + fixedSize; }
  inline constexpr const T* begin() const KJ_LIFETIMEBOUND { return content; }
  inline constexpr const T* end() const KJ_LIFETIMEBOUND { return content + fixedSize; }

  inline constexpr operator ArrayPtr<T>() KJ_LIFETIMEBOUND {
    return arrayPtr(content, fixedSize);
  }
  inline constexpr operator ArrayPtr<const T>() const KJ_LIFETIMEBOUND {
    return arrayPtr(content, fixedSize);
  }

  inline constexpr T& operator[](size_t index) KJ_LIFETIMEBOUND { return content[index]; }
  inline constexpr const T& operator[](size_t index) const KJ_LIFETIMEBOUND {
    return content[index];
  }

private:
  T content[fixedSize];
};

template <typename T, size_t fixedSize>
class CappedArray {
  // Like `FixedArray` but can be dynamically resized as long as the size does not exceed the limit
  // specified by the template parameter.
  //
  // TODO(someday):  Don't construct elements past currentSize?

public:
  inline KJ_CONSTEXPR() CappedArray(): currentSize(fixedSize) {}
  inline explicit constexpr CappedArray(size_t s): currentSize(s) {}

  inline size_t size() const { return currentSize; }
  inline void setSize(size_t s) { KJ_IREQUIRE(s <= fixedSize); currentSize = s; }
  inline T* begin() KJ_LIFETIMEBOUND { return content; }
  inline T* end() KJ_LIFETIMEBOUND { return content + currentSize; }
  inline const T* begin() const KJ_LIFETIMEBOUND { return content; }
  inline const T* end() const KJ_LIFETIMEBOUND { return content + currentSize; }

  inline operator ArrayPtr<T>() KJ_LIFETIMEBOUND {
    return arrayPtr(content, currentSize);
  }
  inline operator ArrayPtr<const T>() const KJ_LIFETIMEBOUND {
    return arrayPtr(content, currentSize);
  }

  inline T& operator[](size_t index) KJ_LIFETIMEBOUND { return content[index]; }
  inline const T& operator[](size_t index) const KJ_LIFETIMEBOUND { return content[index]; }

private:
  size_t currentSize;
  T content[fixedSize];
};

// =======================================================================================
// KJ_MAP

#define KJ_MAP(elementName, array) \
  ::kj::_::Mapper<KJ_DECLTYPE_REF(array)>(array) * \
  [&](typename ::kj::_::Mapper<KJ_DECLTYPE_REF(array)>::Element elementName)
// Applies some function to every element of an array, returning an Array of the results,  with
// nice syntax.  Example:
//
//     StringPtr foo = "abcd";
//     Array<char> bar = KJ_MAP(c, foo) -> char { return c + 1; };
//     KJ_ASSERT(str(bar) == "bcde");

namespace _ {  // private

template <typename T>
struct Mapper {
  T array;
  Mapper(T&& array): array(kj::fwd<T>(array)) {}
  template <typename Func>
  auto operator*(Func&& func) -> Array<decltype(func(*array.begin()))> {
    auto builder = heapArrayBuilder<decltype(func(*array.begin()))>(array.size());
    for (auto iter = array.begin(); iter != array.end(); ++iter) {
      builder.add(func(*iter));
    }
    return builder.finish();
  }
  typedef decltype(*kj::instance<T>().begin()) Element;
};

template <typename T, size_t s>
struct Mapper<T(&)[s]> {
  T* array;
  Mapper(T* array): array(array) {}
  template <typename Func>
  auto operator*(Func&& func) -> Array<decltype(func(*array))> {
    auto builder = heapArrayBuilder<decltype(func(*array))>(s);
    for (size_t i = 0; i < s; i++) {
      builder.add(func(array[i]));
    }
    return builder.finish();
  }
  typedef decltype(*array)& Element;
};

}  // namespace _ (private)

// =======================================================================================
// Inline implementation details

template <typename T>
struct ArrayDisposer::Dispose_<T, true> {
  static void dispose(T* firstElement, size_t elementCount, size_t capacity,
                      const ArrayDisposer& disposer) {
    disposer.disposeImpl(const_cast<RemoveConst<T>*>(firstElement),
                         sizeof(T), elementCount, capacity, nullptr);
  }
};
template <typename T>
struct ArrayDisposer::Dispose_<T, false> {
  static void destruct(void* ptr) {
    kj::dtor(*reinterpret_cast<T*>(ptr));
  }

  static void dispose(T* firstElement, size_t elementCount, size_t capacity,
                      const ArrayDisposer& disposer) {
    disposer.disposeImpl(const_cast<RemoveConst<T>*>(firstElement),
                         sizeof(T), elementCount, capacity, &destruct);
  }
};

template <typename T>
void ArrayDisposer::dispose(T* firstElement, size_t elementCount, size_t capacity) const {
  Dispose_<T>::dispose(firstElement, elementCount, capacity, *this);
}

namespace _ {  // private

template <typename T>
struct HeapArrayDisposer::Allocate_<T, true, true> {
  static T* allocate(size_t elementCount, size_t capacity) {
    return reinterpret_cast<T*>(allocateImpl(
        sizeof(T), elementCount, capacity, nullptr, nullptr));
  }
};
template <typename T>
struct HeapArrayDisposer::Allocate_<T, false, true> {
  static void construct(void* ptr) {
    kj::ctor(*reinterpret_cast<T*>(ptr));
  }
  static T* allocate(size_t elementCount, size_t capacity) {
    return reinterpret_cast<T*>(allocateImpl(
        sizeof(T), elementCount, capacity, &construct, nullptr));
  }
};
template <typename T>
struct HeapArrayDisposer::Allocate_<T, false, false> {
  static void construct(void* ptr) {
    kj::ctor(*reinterpret_cast<T*>(ptr));
  }
  static void destruct(void* ptr) {
    kj::dtor(*reinterpret_cast<T*>(ptr));
  }
  static T* allocate(size_t elementCount, size_t capacity) {
    return reinterpret_cast<T*>(allocateImpl(
        sizeof(T), elementCount, capacity, &construct, &destruct));
  }
};

template <typename T>
T* HeapArrayDisposer::allocate(size_t count) {
  return Allocate_<T>::allocate(count, count);
}

template <typename T>
T* HeapArrayDisposer::allocateUninitialized(size_t count) {
  return Allocate_<T, true, true>::allocate(0, count);
}

template <typename Element, typename Iterator, bool move, bool = canMemcpy<Element>()>
struct CopyConstructArray_;

template <typename T, bool move>
struct CopyConstructArray_<T, T*, move, true> {
  static inline T* apply(T* __restrict__ pos, T* start, T* end) {
    if (end != start) {
      memcpy(pos, start, reinterpret_cast<byte*>(end) - reinterpret_cast<byte*>(start));
    }
    return pos + (end - start);
  }
};

template <typename T>
struct CopyConstructArray_<T, const T*, false, true> {
  static inline T* apply(T* __restrict__ pos, const T* start, const T* end) {
    if (end != start) {
      memcpy(pos, start, reinterpret_cast<const byte*>(end) - reinterpret_cast<const byte*>(start));
    }
    return pos + (end - start);
  }
};

template <typename T, typename Iterator, bool move>
struct CopyConstructArray_<T, Iterator, move, true> {
  static inline T* apply(T* __restrict__ pos, Iterator start, Iterator end) {
    // Since both the copy constructor and assignment operator are trivial, we know that assignment
    // is equivalent to copy-constructing.  So we can make this case somewhat easier for the
    // compiler to optimize.
    while (start != end) {
      *pos++ = *start++;
    }
    return pos;
  }
};

template <typename T, typename Iterator>
struct CopyConstructArray_<T, Iterator, false, false> {
  struct ExceptionGuard {
    T* start;
    T* pos;
    inline explicit ExceptionGuard(T* pos): start(pos), pos(pos) {}
    ~ExceptionGuard() noexcept(false) {
      while (pos > start) {
        dtor(*--pos);
      }
    }
  };

  static T* apply(T* __restrict__ pos, Iterator start, Iterator end) {
    // Verify that T can be *implicitly* constructed from the source values.
    if (false) implicitCast<T>(*start);

    if (noexcept(T(*start))) {
      while (start != end) {
        ctor(*pos++, *start++);
      }
      return pos;
    } else {
      // Crap.  This is complicated.
      ExceptionGuard guard(pos);
      while (start != end) {
        ctor(*guard.pos, *start++);
        ++guard.pos;
      }
      guard.start = guard.pos;
      return guard.pos;
    }
  }
};

template <typename T, typename Iterator>
struct CopyConstructArray_<T, Iterator, true, false> {
  // Actually move-construct.

  struct ExceptionGuard {
    T* start;
    T* pos;
    inline explicit ExceptionGuard(T* pos): start(pos), pos(pos) {}
    ~ExceptionGuard() noexcept(false) {
      while (pos > start) {
        dtor(*--pos);
      }
    }
  };

  static T* apply(T* __restrict__ pos, Iterator start, Iterator end) {
    // Verify that T can be *implicitly* constructed from the source values.
    if (false) implicitCast<T>(kj::mv(*start));

    if (noexcept(T(kj::mv(*start)))) {
      while (start != end) {
        ctor(*pos++, kj::mv(*start++));
      }
      return pos;
    } else {
      // Crap.  This is complicated.
      ExceptionGuard guard(pos);
      while (start != end) {
        ctor(*guard.pos, kj::mv(*start++));
        ++guard.pos;
      }
      guard.start = guard.pos;
      return guard.pos;
    }
  }
};

}  // namespace _ (private)

template <typename T>
template <typename Iterator, bool move>
void ArrayBuilder<T>::addAll(Iterator start, Iterator end) {
  pos = _::CopyConstructArray_<RemoveConst<T>, Decay<Iterator>, move>::apply(pos, start, end);
}

template <typename T>
Array<T> heapArray(const T* content, size_t size) {
  ArrayBuilder<T> builder = heapArrayBuilder<T>(size);
  builder.addAll(content, content + size);
  return builder.finish();
}

template <typename T>
Array<T> heapArray(T* content, size_t size) {
  ArrayBuilder<T> builder = heapArrayBuilder<T>(size);
  builder.addAll(content, content + size);
  return builder.finish();
}

template <typename T>
Array<T> heapArray(ArrayPtr<T> content) {
  ArrayBuilder<T> builder = heapArrayBuilder<T>(content.size());
  builder.addAll(content);
  return builder.finish();
}

template <typename T>
Array<T> heapArray(ArrayPtr<const T> content) {
  ArrayBuilder<T> builder = heapArrayBuilder<T>(content.size());
  builder.addAll(content);
  return builder.finish();
}

template <typename T, typename Iterator> Array<T>
heapArray(Iterator begin, Iterator end) {
  ArrayBuilder<T> builder = heapArrayBuilder<T>(end - begin);
  builder.addAll(begin, end);
  return builder.finish();
}

template <typename T>
inline Array<T> heapArray(std::initializer_list<T> init) {
  return heapArray<T>(init.begin(), init.end());
}

#if KJ_CPP_STD > 201402L
template <typename T, typename... Params>
inline Array<Decay<T>> arr(T&& param1, Params&&... params) {
  ArrayBuilder<Decay<T>> builder = heapArrayBuilder<Decay<T>>(sizeof...(params) + 1);
  (builder.add(kj::fwd<T>(param1)), ... , builder.add(kj::fwd<Params>(params)));
  return builder.finish();
}
template <typename T, typename... Params>
inline Array<Decay<T>> arrOf(Params&&... params) {
  ArrayBuilder<Decay<T>> builder = heapArrayBuilder<Decay<T>>(sizeof...(params));
  (... , builder.add(kj::fwd<Params>(params)));
  return builder.finish();
}
#endif

namespace _ {  // private

template <typename... T>
struct ArrayDisposableOwnedBundle final: public ArrayDisposer, public OwnedBundle<T...> {
  ArrayDisposableOwnedBundle(T&&... values): OwnedBundle<T...>(kj::fwd<T>(values)...) {}
  void disposeImpl(void*, size_t, size_t, size_t, void (*)(void*)) const override { delete this; }
};

}  // namespace _ (private)

template <typename T>
template <typename... Attachments>
Array<T> Array<T>::attach(Attachments&&... attachments) {
  T* ptrCopy = ptr;
  auto sizeCopy = size_;

  KJ_IREQUIRE(ptrCopy != nullptr, "cannot attach to null pointer");

  // HACK: If someone accidentally calls .attach() on a null pointer in opt mode, try our best to
  //   accomplish reasonable behavior: We turn the pointer non-null but still invalid, so that the
  //   disposer will still be called when the pointer goes out of scope.
  if (ptrCopy == nullptr) ptrCopy = reinterpret_cast<T*>(1);

  auto bundle = new _::ArrayDisposableOwnedBundle<Array<T>, Attachments...>(
      kj::mv(*this), kj::fwd<Attachments>(attachments)...);
  return Array<T>(ptrCopy, sizeCopy, *bundle);
}

template <typename T>
template <typename... Attachments>
Array<T> ArrayPtr<T>::attach(Attachments&&... attachments) const {
  T* ptrCopy = ptr;

  KJ_IREQUIRE(ptrCopy != nullptr, "cannot attach to null pointer");

  // HACK: If someone accidentally calls .attach() on a null pointer in opt mode, try our best to
  //   accomplish reasonable behavior: We turn the pointer non-null but still invalid, so that the
  //   disposer will still be called when the pointer goes out of scope.
  if (ptrCopy == nullptr) ptrCopy = reinterpret_cast<T*>(1);

  auto bundle = new _::ArrayDisposableOwnedBundle<Attachments...>(
      kj::fwd<Attachments>(attachments)...);
  return Array<T>(ptrCopy, size_, *bundle);
}

}  // namespace kj

KJ_END_HEADER
