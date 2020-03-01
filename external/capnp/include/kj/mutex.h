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

#ifndef KJ_MUTEX_H_
#define KJ_MUTEX_H_

#if defined(__GNUC__) && !KJ_HEADER_WARNINGS
#pragma GCC system_header
#endif

#include "memory.h"
#include <inttypes.h>

#if __linux__ && !defined(KJ_USE_FUTEX)
#define KJ_USE_FUTEX 1
#endif

#if !KJ_USE_FUTEX && !_WIN32
// On Linux we use futex.  On other platforms we wrap pthreads.
// TODO(someday):  Write efficient low-level locking primitives for other platforms.
#include <pthread.h>
#endif

namespace kj {

// =======================================================================================
// Private details -- public interfaces follow below.

namespace _ {  // private

class Mutex {
  // Internal implementation details.  See `MutexGuarded<T>`.

public:
  Mutex();
  ~Mutex();
  KJ_DISALLOW_COPY(Mutex);

  enum Exclusivity {
    EXCLUSIVE,
    SHARED
  };

  void lock(Exclusivity exclusivity);
  void unlock(Exclusivity exclusivity);

  void assertLockedByCaller(Exclusivity exclusivity);
  // In debug mode, assert that the mutex is locked by the calling thread, or if that is
  // non-trivial, assert that the mutex is locked (which should be good enough to catch problems
  // in unit tests).  In non-debug builds, do nothing.

private:
#if KJ_USE_FUTEX
  uint futex;
  // bit 31 (msb) = set if exclusive lock held
  // bit 30 (msb) = set if threads are waiting for exclusive lock
  // bits 0-29 = count of readers; If an exclusive lock is held, this is the count of threads
  //   waiting for a read lock, otherwise it is the count of threads that currently hold a read
  //   lock.

  static constexpr uint EXCLUSIVE_HELD = 1u << 31;
  static constexpr uint EXCLUSIVE_REQUESTED = 1u << 30;
  static constexpr uint SHARED_COUNT_MASK = EXCLUSIVE_REQUESTED - 1;

#elif _WIN32
  uintptr_t srwLock;  // Actually an SRWLOCK, but don't want to #include <windows.h> in header.

#else
  mutable pthread_rwlock_t mutex;
#endif
};

class Once {
  // Internal implementation details.  See `Lazy<T>`.

public:
#if KJ_USE_FUTEX
  inline Once(bool startInitialized = false)
      : futex(startInitialized ? INITIALIZED : UNINITIALIZED) {}
#else
  Once(bool startInitialized = false);
  ~Once();
#endif
  KJ_DISALLOW_COPY(Once);

  class Initializer {
  public:
    virtual void run() = 0;
  };

  void runOnce(Initializer& init);

#if _WIN32  // TODO(perf): Can we make this inline on win32 somehow?
  bool isInitialized() noexcept;

#else
  inline bool isInitialized() noexcept {
    // Fast path check to see if runOnce() would simply return immediately.
#if KJ_USE_FUTEX
    return __atomic_load_n(&futex, __ATOMIC_ACQUIRE) == INITIALIZED;
#else
    return __atomic_load_n(&state, __ATOMIC_ACQUIRE) == INITIALIZED;
#endif
  }
#endif

  void reset();
  // Returns the state from initialized to uninitialized.  It is an error to call this when
  // not already initialized, or when runOnce() or isInitialized() might be called concurrently in
  // another thread.

private:
#if KJ_USE_FUTEX
  uint futex;

  enum State {
    UNINITIALIZED,
    INITIALIZING,
    INITIALIZING_WITH_WAITERS,
    INITIALIZED
  };

#elif _WIN32
  uintptr_t initOnce;  // Actually an INIT_ONCE, but don't want to #include <windows.h> in header.

#else
  enum State {
    UNINITIALIZED,
    INITIALIZED
  };
  State state;
  pthread_mutex_t mutex;
#endif
};

}  // namespace _ (private)

// =======================================================================================
// Public interface

template <typename T>
class Locked {
  // Return type for `MutexGuarded<T>::lock()`.  `Locked<T>` provides access to the bounded object
  // and unlocks the mutex when it goes out of scope.

public:
  KJ_DISALLOW_COPY(Locked);
  inline Locked(): mutex(nullptr), ptr(nullptr) {}
  inline Locked(Locked&& other): mutex(other.mutex), ptr(other.ptr) {
    other.mutex = nullptr;
    other.ptr = nullptr;
  }
  inline ~Locked() {
    if (mutex != nullptr) mutex->unlock(isConst<T>() ? _::Mutex::SHARED : _::Mutex::EXCLUSIVE);
  }

  inline Locked& operator=(Locked&& other) {
    if (mutex != nullptr) mutex->unlock(isConst<T>() ? _::Mutex::SHARED : _::Mutex::EXCLUSIVE);
    mutex = other.mutex;
    ptr = other.ptr;
    other.mutex = nullptr;
    other.ptr = nullptr;
    return *this;
  }

  inline void release() {
    if (mutex != nullptr) mutex->unlock(isConst<T>() ? _::Mutex::SHARED : _::Mutex::EXCLUSIVE);
    mutex = nullptr;
    ptr = nullptr;
  }

  inline T* operator->() { return ptr; }
  inline const T* operator->() const { return ptr; }
  inline T& operator*() { return *ptr; }
  inline const T& operator*() const { return *ptr; }
  inline T* get() { return ptr; }
  inline const T* get() const { return ptr; }
  inline operator T*() { return ptr; }
  inline operator const T*() const { return ptr; }

private:
  _::Mutex* mutex;
  T* ptr;

  inline Locked(_::Mutex& mutex, T& value): mutex(&mutex), ptr(&value) {}

  template <typename U>
  friend class MutexGuarded;
};

template <typename T>
class MutexGuarded {
  // An object of type T, bounded by a mutex.  In order to access the object, you must lock it.
  //
  // Write locks are not "recursive" -- trying to lock again in a thread that already holds a lock
  // will deadlock.  Recursive write locks are usually a sign of bad design.
  //
  // Unfortunately, **READ LOCKS ARE NOT RECURSIVE** either.  Common sense says they should be.
  // But on many operating systems (BSD, OSX), recursively read-locking a pthread_rwlock is
  // actually unsafe.  The problem is that writers are "prioritized" over readers, so a read lock
  // request will block if any write lock requests are outstanding.  So, if thread A takes a read
  // lock, thread B requests a write lock (and starts waiting), and then thread A tries to take
  // another read lock recursively, the result is deadlock.

public:
  template <typename... Params>
  explicit MutexGuarded(Params&&... params);
  // Initialize the mutex-bounded object by passing the given parameters to its constructor.

  Locked<T> lockExclusive() const;
  // Exclusively locks the object and returns it.  The returned `Locked<T>` can be passed by
  // move, similar to `Own<T>`.
  //
  // This method is declared `const` in accordance with KJ style rules which say that constness
  // should be used to indicate thread-safety.  It is safe to share a const pointer between threads,
  // but it is not safe to share a mutable pointer.  Since the whole point of MutexGuarded is to
  // be shared between threads, its methods should be const, even though locking it produces a
  // non-const pointer to the contained object.

  Locked<const T> lockShared() const;
  // Lock the value for shared access.  Multiple shared locks can be taken concurrently, but cannot
  // be held at the same time as a non-shared lock.

  inline const T& getWithoutLock() const { return value; }
  inline T& getWithoutLock() { return value; }
  // Escape hatch for cases where some external factor guarantees that it's safe to get the
  // value.  You should treat these like const_cast -- be highly suspicious of any use.

  inline const T& getAlreadyLockedShared() const;
  inline T& getAlreadyLockedShared();
  inline T& getAlreadyLockedExclusive() const;
  // Like `getWithoutLock()`, but asserts that the lock is already held by the calling thread.

private:
  mutable _::Mutex mutex;
  mutable T value;
};

template <typename T>
class MutexGuarded<const T> {
  // MutexGuarded cannot guard a const type.  This would be pointless anyway, and would complicate
  // the implementation of Locked<T>, which uses constness to decide what kind of lock it holds.
  static_assert(sizeof(T) < 0, "MutexGuarded's type cannot be const.");
};

template <typename T>
class Lazy {
  // A lazily-initialized value.

public:
  template <typename Func>
  T& get(Func&& init);
  template <typename Func>
  const T& get(Func&& init) const;
  // The first thread to call get() will invoke the given init function to construct the value.
  // Other threads will block until construction completes, then return the same value.
  //
  // `init` is a functor(typically a lambda) which takes `SpaceFor<T>&` as its parameter and returns
  // `Own<T>`.  If `init` throws an exception, the exception is propagated out of that thread's
  // call to `get()`, and subsequent calls behave as if `get()` hadn't been called at all yet --
  // in other words, subsequent calls retry initialization until it succeeds.

private:
  mutable _::Once once;
  mutable SpaceFor<T> space;
  mutable Own<T> value;

  template <typename Func>
  class InitImpl;
};

// =======================================================================================
// Inline implementation details

template <typename T>
template <typename... Params>
inline MutexGuarded<T>::MutexGuarded(Params&&... params)
    : value(kj::fwd<Params>(params)...) {}

template <typename T>
inline Locked<T> MutexGuarded<T>::lockExclusive() const {
  mutex.lock(_::Mutex::EXCLUSIVE);
  return Locked<T>(mutex, value);
}

template <typename T>
inline Locked<const T> MutexGuarded<T>::lockShared() const {
  mutex.lock(_::Mutex::SHARED);
  return Locked<const T>(mutex, value);
}

template <typename T>
inline const T& MutexGuarded<T>::getAlreadyLockedShared() const {
#ifdef KJ_DEBUG
  mutex.assertLockedByCaller(_::Mutex::SHARED);
#endif
  return value;
}
template <typename T>
inline T& MutexGuarded<T>::getAlreadyLockedShared() {
#ifdef KJ_DEBUG
  mutex.assertLockedByCaller(_::Mutex::SHARED);
#endif
  return value;
}
template <typename T>
inline T& MutexGuarded<T>::getAlreadyLockedExclusive() const {
#ifdef KJ_DEBUG
  mutex.assertLockedByCaller(_::Mutex::EXCLUSIVE);
#endif
  return const_cast<T&>(value);
}

template <typename T>
template <typename Func>
class Lazy<T>::InitImpl: public _::Once::Initializer {
public:
  inline InitImpl(const Lazy<T>& lazy, Func&& func): lazy(lazy), func(kj::fwd<Func>(func)) {}

  void run() override {
    lazy.value = func(lazy.space);
  }

private:
  const Lazy<T>& lazy;
  Func func;
};

template <typename T>
template <typename Func>
inline T& Lazy<T>::get(Func&& init) {
  if (!once.isInitialized()) {
    InitImpl<Func> initImpl(*this, kj::fwd<Func>(init));
    once.runOnce(initImpl);
  }
  return *value;
}

template <typename T>
template <typename Func>
inline const T& Lazy<T>::get(Func&& init) const {
  if (!once.isInitialized()) {
    InitImpl<Func> initImpl(*this, kj::fwd<Func>(init));
    once.runOnce(initImpl);
  }
  return *value;
}

}  // namespace kj

#endif  // KJ_MUTEX_H_
