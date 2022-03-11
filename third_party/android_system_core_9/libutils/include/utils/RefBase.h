/*
 * Copyright (C) 2016 The Android Open Source Project
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */


// SOME COMMENTS ABOUT USAGE:

// This provides primarily wp<> weak pointer types and RefBase, which work
// together with sp<> from <StrongPointer.h>.

// sp<> (and wp<>) are a type of smart pointer that use a well defined protocol
// to operate. As long as the object they are templated with implements that
// protocol, these smart pointers work. In several places the platform
// instantiates sp<> with non-RefBase objects; the two are not tied to each
// other.

// RefBase is such an implementation and it supports strong pointers, weak
// pointers and some magic features for the binder.

// So, when using RefBase objects, you have the ability to use strong and weak
// pointers through sp<> and wp<>.

// Normally, when the last strong pointer goes away, the object is destroyed,
// i.e. it's destructor is called. HOWEVER, parts of its associated memory is not
// freed until the last weak pointer is released.

// Weak pointers are essentially "safe" pointers. They are always safe to
// access through promote(). They may return nullptr if the object was
// destroyed because it ran out of strong pointers. This makes them good candidates
// for keys in a cache for instance.

// Weak pointers remain valid for comparison purposes even after the underlying
// object has been destroyed. Even if object A is destroyed and its memory reused
// for B, A remaining weak pointer to A will not compare equal to one to B.
// This again makes them attractive for use as keys.

// How is this supposed / intended to be used?

// Our recommendation is to use strong references (sp<>) when there is an
// ownership relation. e.g. when an object "owns" another one, use a strong
// ref. And of course use strong refs as arguments of functions (it's extremely
// rare that a function will take a wp<>).

// Typically a newly allocated object will immediately be used to initialize
// a strong pointer, which may then be used to construct or assign to other
// strong and weak pointers.

// Use weak references when there are no ownership relation. e.g. the keys in a
// cache (you cannot use plain pointers because there is no safe way to acquire
// a strong reference from a vanilla pointer).

// This implies that two objects should never (or very rarely) have sp<> on
// each other, because they can't both own each other.


// Caveats with reference counting

// Obviously, circular strong references are a big problem; this creates leaks
// and it's hard to debug -- except it's in fact really easy because RefBase has
// tons of debugging code for that. It can basically tell you exactly where the
// leak is.

// Another problem has to do with destructors with side effects. You must
// assume that the destructor of reference counted objects can be called AT ANY
// TIME. For instance code as simple as this:

// void setStuff(const sp<Stuff>& stuff) {
//   std::lock_guard<std::mutex> lock(mMutex);
//   mStuff = stuff;
// }

// is very dangerous. This code WILL deadlock one day or another.

// What isn't obvious is that ~Stuff() can be called as a result of the
// assignment. And it gets called with the lock held. First of all, the lock is
// protecting mStuff, not ~Stuff(). Secondly, if ~Stuff() uses its own internal
// mutex, now you have mutex ordering issues.  Even worse, if ~Stuff() is
// virtual, now you're calling into "user" code (potentially), by that, I mean,
// code you didn't even write.

// A correct way to write this code is something like:

// void setStuff(const sp<Stuff>& stuff) {
//   std::unique_lock<std::mutex> lock(mMutex);
//   sp<Stuff> hold = mStuff;
//   mStuff = stuff;
//   lock.unlock();
// }

// More importantly, reference counted objects should do as little work as
// possible in their destructor, or at least be mindful that their destructor
// could be called from very weird and unintended places.

// Other more specific restrictions for wp<> and sp<>:

// Do not construct a strong pointer to "this" in an object's constructor.
// The onFirstRef() callback would be made on an incompletely constructed
// object.
// Construction of a weak pointer to "this" in an object's constructor is also
// discouraged. But the implementation was recently changed so that, in the
// absence of extendObjectLifetime() calls, weak pointers no longer impact
// object lifetime, and hence this no longer risks premature deallocation,
// and hence usually works correctly.

// Such strong or weak pointers can be safely created in the RefBase onFirstRef()
// callback.

// Use of wp::unsafe_get() for any purpose other than debugging is almost
// always wrong.  Unless you somehow know that there is a longer-lived sp<> to
// the same object, it may well return a pointer to a deallocated object that
// has since been reallocated for a different purpose. (And if you know there
// is a longer-lived sp<>, why not use an sp<> directly?) A wp<> should only be
// dereferenced by using promote().

// Any object inheriting from RefBase should always be destroyed as the result
// of a reference count decrement, not via any other means.  Such objects
// should never be stack allocated, or appear directly as data members in other
// objects. Objects inheriting from RefBase should have their strong reference
// count incremented as soon as possible after construction. Usually this
// will be done via construction of an sp<> to the object, but may instead
// involve other means of calling RefBase::incStrong().
// Explicitly deleting or otherwise destroying a RefBase object with outstanding
// wp<> or sp<> pointers to it will result in an abort or heap corruption.

// It is particularly important not to mix sp<> and direct storage management
// since the sp from raw pointer constructor is implicit. Thus if a RefBase-
// -derived object of type T is managed without ever incrementing its strong
// count, and accidentally passed to f(sp<T>), a strong pointer to the object
// will be temporarily constructed and destroyed, prematurely deallocating the
// object, and resulting in heap corruption. None of this would be easily
// visible in the source.

// Extra Features:

// RefBase::extendObjectLifetime() can be used to prevent destruction of the
// object while there are still weak references. This is really special purpose
// functionality to support Binder.

// Wp::promote(), implemented via the attemptIncStrong() member function, is
// used to try to convert a weak pointer back to a strong pointer.  It's the
// normal way to try to access the fields of an object referenced only through
// a wp<>.  Binder code also sometimes uses attemptIncStrong() directly.

// RefBase provides a number of additional callbacks for certain reference count
// events, as well as some debugging facilities.

// Debugging support can be enabled by turning on DEBUG_REFS in RefBase.cpp.
// Otherwise little checking is provided.

// Thread safety:

// Like std::shared_ptr, sp<> and wp<> allow concurrent accesses to DIFFERENT
// sp<> and wp<> instances that happen to refer to the same underlying object.
// They do NOT support concurrent access (where at least one access is a write)
// to THE SAME sp<> or wp<>.  In effect, their thread-safety properties are
// exactly like those of T*, NOT atomic<T*>.

#ifndef ANDROID_REF_BASE_H
#define ANDROID_REF_BASE_H

#include <atomic>

#include <stdint.h>
#include <sys/types.h>
#include <stdlib.h>
#include <string.h>

// LightRefBase used to be declared in this header, so we have to include it
#include <utils/LightRefBase.h>

#include <utils/StrongPointer.h>
#include <utils/TypeHelpers.h>

// ---------------------------------------------------------------------------
namespace android {

class TextOutput;
TextOutput& printWeakPointer(TextOutput& to, const void* val);

// ---------------------------------------------------------------------------

#define COMPARE_WEAK(_op_)                                      \
inline bool operator _op_ (const sp<T>& o) const {              \
    return m_ptr _op_ o.m_ptr;                                  \
}                                                               \
inline bool operator _op_ (const T* o) const {                  \
    return m_ptr _op_ o;                                        \
}                                                               \
template<typename U>                                            \
inline bool operator _op_ (const sp<U>& o) const {              \
    return m_ptr _op_ o.m_ptr;                                  \
}                                                               \
template<typename U>                                            \
inline bool operator _op_ (const U* o) const {                  \
    return m_ptr _op_ o;                                        \
}

// ---------------------------------------------------------------------------

// RefererenceRenamer is pure abstract, there is no virtual method
// implementation to put in a translation unit in order to silence the
// weak vtables warning.
#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wweak-vtables"
#endif

class ReferenceRenamer {
protected:
    // destructor is purposely not virtual so we avoid code overhead from
    // subclasses; we have to make it protected to guarantee that it
    // cannot be called from this base class (and to make strict compilers
    // happy).
    ~ReferenceRenamer() { }
public:
    virtual void operator()(size_t i) const = 0;
};

#if defined(__clang__)
#pragma clang diagnostic pop
#endif

// ---------------------------------------------------------------------------

class RefBase
{
public:
            void            incStrong(const void* id) const;
            void            decStrong(const void* id) const;
    
            void            forceIncStrong(const void* id) const;

            //! DEBUGGING ONLY: Get current strong ref count.
            int32_t         getStrongCount() const;

    class weakref_type
    {
    public:
        RefBase*            refBase() const;

        void                incWeak(const void* id);
        void                decWeak(const void* id);

        // acquires a strong reference if there is already one.
        bool                attemptIncStrong(const void* id);

        // acquires a weak reference if there is already one.
        // This is not always safe. see ProcessState.cpp and BpBinder.cpp
        // for proper use.
        bool                attemptIncWeak(const void* id);

        //! DEBUGGING ONLY: Get current weak ref count.
        int32_t             getWeakCount() const;

        //! DEBUGGING ONLY: Print references held on object.
        void                printRefs() const;

        //! DEBUGGING ONLY: Enable tracking for this object.
        // enable -- enable/disable tracking
        // retain -- when tracking is enable, if true, then we save a stack trace
        //           for each reference and dereference; when retain == false, we
        //           match up references and dereferences and keep only the
        //           outstanding ones.

        void                trackMe(bool enable, bool retain);
    };

            weakref_type*   createWeak(const void* id) const;
            
            weakref_type*   getWeakRefs() const;

            //! DEBUGGING ONLY: Print references held on object.
    inline  void            printRefs() const { getWeakRefs()->printRefs(); }

            //! DEBUGGING ONLY: Enable tracking of object.
    inline  void            trackMe(bool enable, bool retain)
    { 
        getWeakRefs()->trackMe(enable, retain); 
    }

    typedef RefBase basetype;

protected:
                            RefBase();
    virtual                 ~RefBase();
    
    //! Flags for extendObjectLifetime()
    enum {
        OBJECT_LIFETIME_STRONG  = 0x0000,
        OBJECT_LIFETIME_WEAK    = 0x0001,
        OBJECT_LIFETIME_MASK    = 0x0001
    };
    
            void            extendObjectLifetime(int32_t mode);
            
    //! Flags for onIncStrongAttempted()
    enum {
        FIRST_INC_STRONG = 0x0001
    };
    
    // Invoked after creation of initial strong pointer/reference.
    virtual void            onFirstRef();
    // Invoked when either the last strong reference goes away, or we need to undo
    // the effect of an unnecessary onIncStrongAttempted.
    virtual void            onLastStrongRef(const void* id);
    // Only called in OBJECT_LIFETIME_WEAK case.  Returns true if OK to promote to
    // strong reference. May have side effects if it returns true.
    // The first flags argument is always FIRST_INC_STRONG.
    // TODO: Remove initial flag argument.
    virtual bool            onIncStrongAttempted(uint32_t flags, const void* id);
    // Invoked in the OBJECT_LIFETIME_WEAK case when the last reference of either
    // kind goes away.  Unused.
    // TODO: Remove.
    virtual void            onLastWeakRef(const void* id);

private:
    friend class weakref_type;
    class weakref_impl;
    
                            RefBase(const RefBase& o);
            RefBase&        operator=(const RefBase& o);

private:
    friend class ReferenceMover;

    static void renameRefs(size_t n, const ReferenceRenamer& renamer);

    static void renameRefId(weakref_type* ref,
            const void* old_id, const void* new_id);

    static void renameRefId(RefBase* ref,
            const void* old_id, const void* new_id);

        weakref_impl* const mRefs;
};

// ---------------------------------------------------------------------------

template <typename T>
class wp
{
public:
    typedef typename RefBase::weakref_type weakref_type;

    inline wp() : m_ptr(0) { }

    wp(T* other);  // NOLINT(implicit)
    wp(const wp<T>& other);
    explicit wp(const sp<T>& other);
    template<typename U> wp(U* other);  // NOLINT(implicit)
    template<typename U> wp(const sp<U>& other);  // NOLINT(implicit)
    template<typename U> wp(const wp<U>& other);  // NOLINT(implicit)

    ~wp();

    // Assignment

    wp& operator = (T* other);
    wp& operator = (const wp<T>& other);
    wp& operator = (const sp<T>& other);

    template<typename U> wp& operator = (U* other);
    template<typename U> wp& operator = (const wp<U>& other);
    template<typename U> wp& operator = (const sp<U>& other);

    void set_object_and_refs(T* other, weakref_type* refs);

    // promotion to sp

    sp<T> promote() const;

    // Reset

    void clear();

    // Accessors

    inline  weakref_type* get_refs() const { return m_refs; }

    inline  T* unsafe_get() const { return m_ptr; }

    // Operators

    COMPARE_WEAK(==)
    COMPARE_WEAK(!=)
    COMPARE_WEAK(>)
    COMPARE_WEAK(<)
    COMPARE_WEAK(<=)
    COMPARE_WEAK(>=)

    inline bool operator == (const wp<T>& o) const {
        return (m_ptr == o.m_ptr) && (m_refs == o.m_refs);
    }
    template<typename U>
    inline bool operator == (const wp<U>& o) const {
        return m_ptr == o.m_ptr;
    }

    inline bool operator > (const wp<T>& o) const {
        return (m_ptr == o.m_ptr) ? (m_refs > o.m_refs) : (m_ptr > o.m_ptr);
    }
    template<typename U>
    inline bool operator > (const wp<U>& o) const {
        return (m_ptr == o.m_ptr) ? (m_refs > o.m_refs) : (m_ptr > o.m_ptr);
    }

    inline bool operator < (const wp<T>& o) const {
        return (m_ptr == o.m_ptr) ? (m_refs < o.m_refs) : (m_ptr < o.m_ptr);
    }
    template<typename U>
    inline bool operator < (const wp<U>& o) const {
        return (m_ptr == o.m_ptr) ? (m_refs < o.m_refs) : (m_ptr < o.m_ptr);
    }
                         inline bool operator != (const wp<T>& o) const { return m_refs != o.m_refs; }
    template<typename U> inline bool operator != (const wp<U>& o) const { return !operator == (o); }
                         inline bool operator <= (const wp<T>& o) const { return !operator > (o); }
    template<typename U> inline bool operator <= (const wp<U>& o) const { return !operator > (o); }
                         inline bool operator >= (const wp<T>& o) const { return !operator < (o); }
    template<typename U> inline bool operator >= (const wp<U>& o) const { return !operator < (o); }

private:
    template<typename Y> friend class sp;
    template<typename Y> friend class wp;

    T*              m_ptr;
    weakref_type*   m_refs;
};

template <typename T>
TextOutput& operator<<(TextOutput& to, const wp<T>& val);

#undef COMPARE_WEAK

// ---------------------------------------------------------------------------
// No user serviceable parts below here.

template<typename T>
wp<T>::wp(T* other)
    : m_ptr(other)
{
    if (other) m_refs = other->createWeak(this);
}

template<typename T>
wp<T>::wp(const wp<T>& other)
    : m_ptr(other.m_ptr), m_refs(other.m_refs)
{
    if (m_ptr) m_refs->incWeak(this);
}

template<typename T>
wp<T>::wp(const sp<T>& other)
    : m_ptr(other.m_ptr)
{
    if (m_ptr) {
        m_refs = m_ptr->createWeak(this);
    }
}

template<typename T> template<typename U>
wp<T>::wp(U* other)
    : m_ptr(other)
{
    if (other) m_refs = other->createWeak(this);
}

template<typename T> template<typename U>
wp<T>::wp(const wp<U>& other)
    : m_ptr(other.m_ptr)
{
    if (m_ptr) {
        m_refs = other.m_refs;
        m_refs->incWeak(this);
    }
}

template<typename T> template<typename U>
wp<T>::wp(const sp<U>& other)
    : m_ptr(other.m_ptr)
{
    if (m_ptr) {
        m_refs = m_ptr->createWeak(this);
    }
}

template<typename T>
wp<T>::~wp()
{
    if (m_ptr) m_refs->decWeak(this);
}

template<typename T>
wp<T>& wp<T>::operator = (T* other)
{
    weakref_type* newRefs =
        other ? other->createWeak(this) : 0;
    if (m_ptr) m_refs->decWeak(this);
    m_ptr = other;
    m_refs = newRefs;
    return *this;
}

template<typename T>
wp<T>& wp<T>::operator = (const wp<T>& other)
{
    weakref_type* otherRefs(other.m_refs);
    T* otherPtr(other.m_ptr);
    if (otherPtr) otherRefs->incWeak(this);
    if (m_ptr) m_refs->decWeak(this);
    m_ptr = otherPtr;
    m_refs = otherRefs;
    return *this;
}

template<typename T>
wp<T>& wp<T>::operator = (const sp<T>& other)
{
    weakref_type* newRefs =
        other != NULL ? other->createWeak(this) : 0;
    T* otherPtr(other.m_ptr);
    if (m_ptr) m_refs->decWeak(this);
    m_ptr = otherPtr;
    m_refs = newRefs;
    return *this;
}

template<typename T> template<typename U>
wp<T>& wp<T>::operator = (U* other)
{
    weakref_type* newRefs =
        other ? other->createWeak(this) : 0;
    if (m_ptr) m_refs->decWeak(this);
    m_ptr = other;
    m_refs = newRefs;
    return *this;
}

template<typename T> template<typename U>
wp<T>& wp<T>::operator = (const wp<U>& other)
{
    weakref_type* otherRefs(other.m_refs);
    U* otherPtr(other.m_ptr);
    if (otherPtr) otherRefs->incWeak(this);
    if (m_ptr) m_refs->decWeak(this);
    m_ptr = otherPtr;
    m_refs = otherRefs;
    return *this;
}

template<typename T> template<typename U>
wp<T>& wp<T>::operator = (const sp<U>& other)
{
    weakref_type* newRefs =
        other != NULL ? other->createWeak(this) : 0;
    U* otherPtr(other.m_ptr);
    if (m_ptr) m_refs->decWeak(this);
    m_ptr = otherPtr;
    m_refs = newRefs;
    return *this;
}

template<typename T>
void wp<T>::set_object_and_refs(T* other, weakref_type* refs)
{
    if (other) refs->incWeak(this);
    if (m_ptr) m_refs->decWeak(this);
    m_ptr = other;
    m_refs = refs;
}

template<typename T>
sp<T> wp<T>::promote() const
{
    sp<T> result;
    if (m_ptr && m_refs->attemptIncStrong(&result)) {
        result.set_pointer(m_ptr);
    }
    return result;
}

template<typename T>
void wp<T>::clear()
{
    if (m_ptr) {
        m_refs->decWeak(this);
        m_ptr = 0;
    }
}

template <typename T>
inline TextOutput& operator<<(TextOutput& to, const wp<T>& val)
{
    return printWeakPointer(to, val.unsafe_get());
}

// ---------------------------------------------------------------------------

// this class just serves as a namespace so TYPE::moveReferences can stay
// private.
class ReferenceMover {
public:
    // it would be nice if we could make sure no extra code is generated
    // for sp<TYPE> or wp<TYPE> when TYPE is a descendant of RefBase:
    // Using a sp<RefBase> override doesn't work; it's a bit like we wanted
    // a template<typename TYPE inherits RefBase> template...

    template<typename TYPE> static inline
    void move_references(sp<TYPE>* dest, sp<TYPE> const* src, size_t n) {

        class Renamer : public ReferenceRenamer {
            sp<TYPE>* d_;
            sp<TYPE> const* s_;
            virtual void operator()(size_t i) const {
                // The id are known to be the sp<>'s this pointer
                TYPE::renameRefId(d_[i].get(), &s_[i], &d_[i]);
            }
        public:
            Renamer(sp<TYPE>* d, sp<TYPE> const* s) : d_(d), s_(s) { }
            virtual ~Renamer() { }
        };

        memmove(dest, src, n*sizeof(sp<TYPE>));
        TYPE::renameRefs(n, Renamer(dest, src));
    }


    template<typename TYPE> static inline
    void move_references(wp<TYPE>* dest, wp<TYPE> const* src, size_t n) {

        class Renamer : public ReferenceRenamer {
            wp<TYPE>* d_;
            wp<TYPE> const* s_;
            virtual void operator()(size_t i) const {
                // The id are known to be the wp<>'s this pointer
                TYPE::renameRefId(d_[i].get_refs(), &s_[i], &d_[i]);
            }
        public:
            Renamer(wp<TYPE>* rd, wp<TYPE> const* rs) : d_(rd), s_(rs) { }
            virtual ~Renamer() { }
        };

        memmove(dest, src, n*sizeof(wp<TYPE>));
        TYPE::renameRefs(n, Renamer(dest, src));
    }
};

// specialization for moving sp<> and wp<> types.
// these are used by the [Sorted|Keyed]Vector<> implementations
// sp<> and wp<> need to be handled specially, because they do not
// have trivial copy operation in the general case (see RefBase.cpp
// when DEBUG ops are enabled), but can be implemented very
// efficiently in most cases.

template<typename TYPE> inline
void move_forward_type(sp<TYPE>* d, sp<TYPE> const* s, size_t n) {
    ReferenceMover::move_references(d, s, n);
}

template<typename TYPE> inline
void move_backward_type(sp<TYPE>* d, sp<TYPE> const* s, size_t n) {
    ReferenceMover::move_references(d, s, n);
}

template<typename TYPE> inline
void move_forward_type(wp<TYPE>* d, wp<TYPE> const* s, size_t n) {
    ReferenceMover::move_references(d, s, n);
}

template<typename TYPE> inline
void move_backward_type(wp<TYPE>* d, wp<TYPE> const* s, size_t n) {
    ReferenceMover::move_references(d, s, n);
}

}; // namespace android

// ---------------------------------------------------------------------------

#endif // ANDROID_REF_BASE_H
