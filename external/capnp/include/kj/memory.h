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

#ifndef KJ_MEMORY_H_
#define KJ_MEMORY_H_

#if defined(__GNUC__) && !KJ_HEADER_WARNINGS
#pragma GCC system_header
#endif

#include "common.h"

namespace kj {

// =======================================================================================
// Disposer -- Implementation details.

class Disposer {
  // Abstract interface for a thing that "disposes" of objects, where "disposing" usually means
  // calling the destructor followed by freeing the underlying memory.  `Own<T>` encapsulates an
  // object pointer with corresponding Disposer.
  //
  // Few developers will ever touch this interface.  It is primarily useful for those implementing
  // custom memory allocators.

protected:
  // Do not declare a destructor, as doing so will force a global initializer for each HeapDisposer
  // instance.  Eww!

  virtual void disposeImpl(void* pointer) const = 0;
  // Disposes of the object, given a pointer to the beginning of the object.  If the object is
  // polymorphic, this pointer is determined by dynamic_cast<void*>().  For non-polymorphic types,
  // Own<T> does not allow any casting, so the pointer exactly matches the original one given to
  // Own<T>.

public:

  template <typename T>
  void dispose(T* object) const;
  // Helper wrapper around disposeImpl().
  //
  // If T is polymorphic, calls `disposeImpl(dynamic_cast<void*>(object))`, otherwise calls
  // `disposeImpl(implicitCast<void*>(object))`.
  //
  // Callers must not call dispose() on the same pointer twice, even if the first call throws
  // an exception.

private:
  template <typename T, bool polymorphic = __is_polymorphic(T)>
  struct Dispose_;
};

template <typename T>
class DestructorOnlyDisposer: public Disposer {
  // A disposer that merely calls the type's destructor and nothing else.

public:
  static const DestructorOnlyDisposer instance;

  void disposeImpl(void* pointer) const override {
    reinterpret_cast<T*>(pointer)->~T();
  }
};

template <typename T>
const DestructorOnlyDisposer<T> DestructorOnlyDisposer<T>::instance = DestructorOnlyDisposer<T>();

class NullDisposer: public Disposer {
  // A disposer that does nothing.

public:
  static const NullDisposer instance;

  void disposeImpl(void* pointer) const override {}
};

// =======================================================================================
// Own<T> -- An owned pointer.

template <typename T>
class Own {
  // A transferrable title to a T.  When an Own<T> goes out of scope, the object's Disposer is
  // called to dispose of it.  An Own<T> can be efficiently passed by move, without relocating the
  // underlying object; this transfers ownership.
  //
  // This is much like std::unique_ptr, except:
  // - You cannot release().  An owned object is not necessarily allocated with new (see next
  //   point), so it would be hard to use release() correctly.
  // - The deleter is made polymorphic by virtual call rather than by template.  This is much
  //   more powerful -- it allows the use of custom allocators, freelists, etc.  This could
  //   _almost_ be accomplished with unique_ptr by forcing everyone to use something like
  //   std::unique_ptr<T, kj::Deleter>, except that things get hairy in the presence of multiple
  //   inheritance and upcasting, and anyway if you force everyone to use a custom deleter
  //   then you've lost any benefit to interoperating with the "standard" unique_ptr.

public:
  KJ_DISALLOW_COPY(Own);
  inline Own(): disposer(nullptr), ptr(nullptr) {}
  inline Own(Own&& other) noexcept
      : disposer(other.disposer), ptr(other.ptr) { other.ptr = nullptr; }
  inline Own(Own<RemoveConstOrDisable<T>>&& other) noexcept
      : disposer(other.disposer), ptr(other.ptr) { other.ptr = nullptr; }
  template <typename U, typename = EnableIf<canConvert<U*, T*>()>>
  inline Own(Own<U>&& other) noexcept
      : disposer(other.disposer), ptr(other.ptr) {
    static_assert(__is_polymorphic(T),
        "Casting owned pointers requires that the target type is polymorphic.");
    other.ptr = nullptr;
  }
  inline Own(T* ptr, const Disposer& disposer) noexcept: disposer(&disposer), ptr(ptr) {}

  ~Own() noexcept(false) { dispose(); }

  inline Own& operator=(Own&& other) {
    // Move-assingnment operator.

    // Careful, this might own `other`.  Therefore we have to transfer the pointers first, then
    // dispose.
    const Disposer* disposerCopy = disposer;
    T* ptrCopy = ptr;
    disposer = other.disposer;
    ptr = other.ptr;
    other.ptr = nullptr;
    if (ptrCopy != nullptr) {
      disposerCopy->dispose(const_cast<RemoveConst<T>*>(ptrCopy));
    }
    return *this;
  }

  inline Own& operator=(decltype(nullptr)) {
    dispose();
    return *this;
  }

  template <typename U>
  Own<U> downcast() {
    // Downcast the pointer to Own<U>, destroying the original pointer.  If this pointer does not
    // actually point at an instance of U, the results are undefined (throws an exception in debug
    // mode if RTTI is enabled, otherwise you're on your own).

    Own<U> result;
    if (ptr != nullptr) {
      result.ptr = &kj::downcast<U>(*ptr);
      result.disposer = disposer;
      ptr = nullptr;
    }
    return result;
  }

#define NULLCHECK KJ_IREQUIRE(ptr != nullptr, "null Own<> dereference")
  inline T* operator->() { NULLCHECK; return ptr; }
  inline const T* operator->() const { NULLCHECK; return ptr; }
  inline T& operator*() { NULLCHECK; return *ptr; }
  inline const T& operator*() const { NULLCHECK; return *ptr; }
#undef NULLCHECK
  inline T* get() { return ptr; }
  inline const T* get() const { return ptr; }
  inline operator T*() { return ptr; }
  inline operator const T*() const { return ptr; }

private:
  const Disposer* disposer;  // Only valid if ptr != nullptr.
  T* ptr;

  inline explicit Own(decltype(nullptr)): disposer(nullptr), ptr(nullptr) {}

  inline bool operator==(decltype(nullptr)) { return ptr == nullptr; }
  inline bool operator!=(decltype(nullptr)) { return ptr != nullptr; }
  // Only called by Maybe<Own<T>>.

  inline void dispose() {
    // Make sure that if an exception is thrown, we are left with a null ptr, so we won't possibly
    // dispose again.
    T* ptrCopy = ptr;
    if (ptrCopy != nullptr) {
      ptr = nullptr;
      disposer->dispose(const_cast<RemoveConst<T>*>(ptrCopy));
    }
  }

  template <typename U>
  friend class Own;
  friend class Maybe<Own<T>>;
};

namespace _ {  // private

template <typename T>
class OwnOwn {
public:
  inline OwnOwn(Own<T>&& value) noexcept: value(kj::mv(value)) {}

  inline Own<T>& operator*() & { return value; }
  inline const Own<T>& operator*() const & { return value; }
  inline Own<T>&& operator*() && { return kj::mv(value); }
  inline const Own<T>&& operator*() const && { return kj::mv(value); }
  inline Own<T>* operator->() { return &value; }
  inline const Own<T>* operator->() const { return &value; }
  inline operator Own<T>*() { return value ? &value : nullptr; }
  inline operator const Own<T>*() const { return value ? &value : nullptr; }

private:
  Own<T> value;
};

template <typename T>
OwnOwn<T> readMaybe(Maybe<Own<T>>&& maybe) { return OwnOwn<T>(kj::mv(maybe.ptr)); }
template <typename T>
Own<T>* readMaybe(Maybe<Own<T>>& maybe) { return maybe.ptr ? &maybe.ptr : nullptr; }
template <typename T>
const Own<T>* readMaybe(const Maybe<Own<T>>& maybe) { return maybe.ptr ? &maybe.ptr : nullptr; }

}  // namespace _ (private)

template <typename T>
class Maybe<Own<T>> {
public:
  inline Maybe(): ptr(nullptr) {}
  inline Maybe(Own<T>&& t) noexcept: ptr(kj::mv(t)) {}
  inline Maybe(Maybe&& other) noexcept: ptr(kj::mv(other.ptr)) {}

  template <typename U>
  inline Maybe(Maybe<Own<U>>&& other): ptr(mv(other.ptr)) {}
  template <typename U>
  inline Maybe(Own<U>&& other): ptr(mv(other)) {}

  inline Maybe(decltype(nullptr)) noexcept: ptr(nullptr) {}

  inline operator Maybe<T&>() { return ptr.get(); }
  inline operator Maybe<const T&>() const { return ptr.get(); }

  inline Maybe& operator=(Maybe&& other) { ptr = kj::mv(other.ptr); return *this; }

  inline bool operator==(decltype(nullptr)) const { return ptr == nullptr; }
  inline bool operator!=(decltype(nullptr)) const { return ptr != nullptr; }

  Own<T>& orDefault(Own<T>& defaultValue) {
    if (ptr == nullptr) {
      return defaultValue;
    } else {
      return ptr;
    }
  }
  const Own<T>& orDefault(const Own<T>& defaultValue) const {
    if (ptr == nullptr) {
      return defaultValue;
    } else {
      return ptr;
    }
  }

  template <typename Func>
  auto map(Func&& f) & -> Maybe<decltype(f(instance<Own<T>&>()))> {
    if (ptr == nullptr) {
      return nullptr;
    } else {
      return f(ptr);
    }
  }

  template <typename Func>
  auto map(Func&& f) const & -> Maybe<decltype(f(instance<const Own<T>&>()))> {
    if (ptr == nullptr) {
      return nullptr;
    } else {
      return f(ptr);
    }
  }

  template <typename Func>
  auto map(Func&& f) && -> Maybe<decltype(f(instance<Own<T>&&>()))> {
    if (ptr == nullptr) {
      return nullptr;
    } else {
      return f(kj::mv(ptr));
    }
  }

  template <typename Func>
  auto map(Func&& f) const && -> Maybe<decltype(f(instance<const Own<T>&&>()))> {
    if (ptr == nullptr) {
      return nullptr;
    } else {
      return f(kj::mv(ptr));
    }
  }

private:
  Own<T> ptr;

  template <typename U>
  friend class Maybe;
  template <typename U>
  friend _::OwnOwn<U> _::readMaybe(Maybe<Own<U>>&& maybe);
  template <typename U>
  friend Own<U>* _::readMaybe(Maybe<Own<U>>& maybe);
  template <typename U>
  friend const Own<U>* _::readMaybe(const Maybe<Own<U>>& maybe);
};

namespace _ {  // private

template <typename T>
class HeapDisposer final: public Disposer {
public:
  virtual void disposeImpl(void* pointer) const override { delete reinterpret_cast<T*>(pointer); }

  static const HeapDisposer instance;
};

template <typename T>
const HeapDisposer<T> HeapDisposer<T>::instance = HeapDisposer<T>();

}  // namespace _ (private)

template <typename T, typename... Params>
Own<T> heap(Params&&... params) {
  // heap<T>(...) allocates a T on the heap, forwarding the parameters to its constructor.  The
  // exact heap implementation is unspecified -- for now it is operator new, but you should not
  // assume this.  (Since we know the object size at delete time, we could actually implement an
  // allocator that is more efficient than operator new.)

  return Own<T>(new T(kj::fwd<Params>(params)...), _::HeapDisposer<T>::instance);
}

template <typename T>
Own<Decay<T>> heap(T&& orig) {
  // Allocate a copy (or move) of the argument on the heap.
  //
  // The purpose of this overload is to allow you to omit the template parameter as there is only
  // one argument and the purpose is to copy it.

  typedef Decay<T> T2;
  return Own<T2>(new T2(kj::fwd<T>(orig)), _::HeapDisposer<T2>::instance);
}

// =======================================================================================
// SpaceFor<T> -- assists in manual allocation

template <typename T>
class SpaceFor {
  // A class which has the same size and alignment as T but does not call its constructor or
  // destructor automatically.  Instead, call construct() to construct a T in the space, which
  // returns an Own<T> which will take care of calling T's destructor later.

public:
  inline SpaceFor() {}
  inline ~SpaceFor() {}

  template <typename... Params>
  Own<T> construct(Params&&... params) {
    ctor(value, kj::fwd<Params>(params)...);
    return Own<T>(&value, DestructorOnlyDisposer<T>::instance);
  }

private:
  union {
    T value;
  };
};

// =======================================================================================
// Inline implementation details

template <typename T>
struct Disposer::Dispose_<T, true> {
  static void dispose(T* object, const Disposer& disposer) {
    // Note that dynamic_cast<void*> does not require RTTI to be enabled, because the offset to
    // the top of the object is in the vtable -- as it obviously needs to be to correctly implement
    // operator delete.
    disposer.disposeImpl(dynamic_cast<void*>(object));
  }
};
template <typename T>
struct Disposer::Dispose_<T, false> {
  static void dispose(T* object, const Disposer& disposer) {
    disposer.disposeImpl(static_cast<void*>(object));
  }
};

template <typename T>
void Disposer::dispose(T* object) const {
  Dispose_<T>::dispose(object, *this);
}

}  // namespace kj

#endif  // KJ_MEMORY_H_
