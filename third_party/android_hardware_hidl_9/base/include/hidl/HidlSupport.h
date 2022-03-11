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

#ifndef ANDROID_HIDL_SUPPORT_H
#define ANDROID_HIDL_SUPPORT_H

#include <algorithm>
#include <array>
#include <iterator>
#include <cutils/native_handle.h>
#include <hidl/HidlInternal.h>
#include <hidl/Status.h>
#include <map>
#include <sstream>
#include <stddef.h>
#include <tuple>
#include <type_traits>
#include <utils/Errors.h>
#include <utils/RefBase.h>
#include <utils/StrongPointer.h>
#include <vector>

namespace android {

// this file is included by all hidl interface, so we must forward declare the
// IMemory and IBase types.
namespace hidl {
namespace memory {
namespace V1_0 {
    struct IMemory;
}; // namespace V1_0
}; // namespace manager
}; // namespace hidl

namespace hidl {
namespace base {
namespace V1_0 {
    struct IBase;
}; // namespace V1_0
}; // namespace base
}; // namespace hidl

namespace hardware {

namespace details {
// Return true on userdebug / eng builds and false on user builds.
bool debuggable();
} //  namespace details

// hidl_death_recipient is a callback interfaced that can be used with
// linkToDeath() / unlinkToDeath()
struct hidl_death_recipient : public virtual RefBase {
    virtual void serviceDied(uint64_t cookie,
            const ::android::wp<::android::hidl::base::V1_0::IBase>& who) = 0;
};

// hidl_handle wraps a pointer to a native_handle_t in a hidl_pointer,
// so that it can safely be transferred between 32-bit and 64-bit processes.
// The ownership semantics for this are:
// 1) The conversion constructor and assignment operator taking a const native_handle_t*
//    do not take ownership of the handle; this is because these operations are usually
//    just done for IPC, and cloning by default is a waste of resources. If you want
//    a hidl_handle to take ownership, call setTo(handle, true /*shouldOwn*/);
// 2) The copy constructor/assignment operator taking a hidl_handle *DO* take ownership;
//    that is because it's not intuitive that this class encapsulates a native_handle_t
//    which needs cloning to be valid; in particular, this allows constructs like this:
//    hidl_handle copy;
//    foo->someHidlCall([&](auto incoming_handle) {
//            copy = incoming_handle;
//    });
//    // copy and its enclosed file descriptors will remain valid here.
// 3) The move constructor does what you would expect; it only owns the handle if the
//    original did.
struct hidl_handle {
    hidl_handle();
    ~hidl_handle();

    hidl_handle(const native_handle_t *handle);

    // copy constructor.
    hidl_handle(const hidl_handle &other);

    // move constructor.
    hidl_handle(hidl_handle &&other) noexcept;

    // assignment operators
    hidl_handle &operator=(const hidl_handle &other);

    hidl_handle &operator=(const native_handle_t *native_handle);

    hidl_handle &operator=(hidl_handle &&other) noexcept;

    void setTo(native_handle_t* handle, bool shouldOwn = false);

    const native_handle_t* operator->() const;

    // implicit conversion to const native_handle_t*
    operator const native_handle_t *() const;

    // explicit conversion
    const native_handle_t *getNativeHandle() const;
private:
    void freeHandle();

    details::hidl_pointer<const native_handle_t> mHandle __attribute__ ((aligned(8)));
    bool mOwnsHandle __attribute ((aligned(8)));
};

struct hidl_string {
    hidl_string();
    ~hidl_string();

    // copy constructor.
    hidl_string(const hidl_string &);
    // copy from a C-style string. nullptr will create an empty string
    hidl_string(const char *);
    // copy the first length characters from a C-style string.
    hidl_string(const char *, size_t length);
    // copy from an std::string.
    hidl_string(const std::string &);

    // move constructor.
    hidl_string(hidl_string &&) noexcept;

    const char *c_str() const;
    size_t size() const;
    bool empty() const;

    // copy assignment operator.
    hidl_string &operator=(const hidl_string &);
    // copy from a C-style string.
    hidl_string &operator=(const char *s);
    // copy from an std::string.
    hidl_string &operator=(const std::string &);
    // move assignment operator.
    hidl_string &operator=(hidl_string &&other) noexcept;
    // cast to std::string.
    operator std::string() const;

    void clear();

    // Reference an external char array. Ownership is _not_ transferred.
    // Caller is responsible for ensuring that underlying memory is valid
    // for the lifetime of this hidl_string.
    void setToExternal(const char *data, size_t size);

    // offsetof(hidl_string, mBuffer) exposed since mBuffer is private.
    static const size_t kOffsetOfBuffer;

private:
    details::hidl_pointer<const char> mBuffer;
    uint32_t mSize;  // NOT including the terminating '\0'.
    bool mOwnsBuffer; // if true then mBuffer is a mutable char *

    // copy from data with size. Assume that my memory is freed
    // (through clear(), for example)
    void copyFrom(const char *data, size_t size);
    // move from another hidl_string
    void moveFrom(hidl_string &&);
};

// Use NOLINT to suppress missing parentheses warnings around OP.
#define HIDL_STRING_OPERATOR(OP)                                              \
    inline bool operator OP(const hidl_string& hs1, const hidl_string& hs2) { \
        return strcmp(hs1.c_str(), hs2.c_str()) OP 0; /* NOLINT */            \
    }                                                                         \
    inline bool operator OP(const hidl_string& hs, const char* s) {           \
        return strcmp(hs.c_str(), s) OP 0; /* NOLINT */                       \
    }                                                                         \
    inline bool operator OP(const char* s, const hidl_string& hs) {           \
        return strcmp(s, hs.c_str()) OP 0; /* NOLINT */                       \
    }

HIDL_STRING_OPERATOR(==)
HIDL_STRING_OPERATOR(!=)
HIDL_STRING_OPERATOR(<)
HIDL_STRING_OPERATOR(<=)
HIDL_STRING_OPERATOR(>)
HIDL_STRING_OPERATOR(>=)

#undef HIDL_STRING_OPERATOR

// Send our content to the output stream
std::ostream& operator<<(std::ostream& os, const hidl_string& str);


// hidl_memory is a structure that can be used to transfer
// pieces of shared memory between processes. The assumption
// of this object is that the memory remains accessible as
// long as the file descriptors in the enclosed mHandle
// - as well as all of its cross-process dups() - remain opened.
struct hidl_memory {

    hidl_memory() : mHandle(nullptr), mSize(0), mName("") {
    }

    /**
     * Creates a hidl_memory object whose handle has the same lifetime
     * as the handle moved into it.
     */
    hidl_memory(const hidl_string& name, hidl_handle&& handle, size_t size)
        : mHandle(std::move(handle)), mSize(size), mName(name) {}

    /**
     * Creates a hidl_memory object, but doesn't take ownership of
     * the passed in native_handle_t; callers are responsible for
     * making sure the handle remains valid while this object is
     * used.
     */
    hidl_memory(const hidl_string &name, const native_handle_t *handle, size_t size)
      :  mHandle(handle),
         mSize(size),
         mName(name)
    {}

    // copy constructor
    hidl_memory(const hidl_memory& other) {
        *this = other;
    }

    // copy assignment
    hidl_memory &operator=(const hidl_memory &other) {
        if (this != &other) {
            mHandle = other.mHandle;
            mSize = other.mSize;
            mName = other.mName;
        }

        return *this;
    }

    // move constructor
    hidl_memory(hidl_memory&& other) noexcept {
        *this = std::move(other);
    }

    // move assignment
    hidl_memory &operator=(hidl_memory &&other) noexcept {
        if (this != &other) {
            mHandle = std::move(other.mHandle);
            mSize = other.mSize;
            mName = std::move(other.mName);
            other.mSize = 0;
        }

        return *this;
    }


    ~hidl_memory() {
    }

    const native_handle_t* handle() const {
        return mHandle;
    }

    const hidl_string &name() const {
        return mName;
    }

    uint64_t size() const {
        return mSize;
    }

    // @return true if it's valid
    inline bool valid() const { return handle() != nullptr; }

    // offsetof(hidl_memory, mHandle) exposed since mHandle is private.
    static const size_t kOffsetOfHandle;
    // offsetof(hidl_memory, mName) exposed since mHandle is private.
    static const size_t kOffsetOfName;

private:
    hidl_handle mHandle __attribute__ ((aligned(8)));
    uint64_t mSize __attribute__ ((aligned(8)));
    hidl_string mName __attribute__ ((aligned(8)));
};

// HidlMemory is a wrapper class to support sp<> for hidl_memory. It also
// provides factory methods to create an instance from hidl_memory or
// from a opened file descriptor. The number of factory methods can be increase
// to support other type of hidl_memory without break the ABI.
class HidlMemory : public virtual hidl_memory, public virtual ::android::RefBase {
public:
    static sp<HidlMemory> getInstance(const hidl_memory& mem);

    static sp<HidlMemory> getInstance(hidl_memory&& mem);

    static sp<HidlMemory> getInstance(const hidl_string& name, hidl_handle&& handle, uint64_t size);
    // @param fd, shall be opened and points to the resource.
    // @note this method takes the ownership of the fd and will close it in
    //     destructor
    // @return nullptr in failure with the fd closed
    static sp<HidlMemory> getInstance(const hidl_string& name, int fd, uint64_t size);

    virtual ~HidlMemory();

protected:
    HidlMemory();
    HidlMemory(const hidl_string& name, hidl_handle&& handle, size_t size);
};
////////////////////////////////////////////////////////////////////////////////

template<typename T>
struct hidl_vec {
    hidl_vec() {
        static_assert(hidl_vec<T>::kOffsetOfBuffer == 0, "wrong offset");

        memset(this, 0, sizeof(*this));
        // mSize is 0
        // mBuffer is nullptr

        // this is for consistency with the original implementation
        mOwnsBuffer = true;
    }

    // Note, does not initialize primitive types.
    hidl_vec(size_t size) : hidl_vec() { resize(size); }

    hidl_vec(const hidl_vec<T> &other) : hidl_vec() {
        *this = other;
    }

    hidl_vec(hidl_vec<T> &&other) noexcept : hidl_vec() {
        *this = std::move(other);
    }

    hidl_vec(const std::initializer_list<T> list) : hidl_vec() {
        if (list.size() > UINT32_MAX) {
            details::logAlwaysFatal("hidl_vec can't hold more than 2^32 elements.");
        }
        mSize = static_cast<uint32_t>(list.size());
        mBuffer = new T[mSize]();
        mOwnsBuffer = true;

        size_t idx = 0;
        for (auto it = list.begin(); it != list.end(); ++it) {
            mBuffer[idx++] = *it;
        }
    }

    hidl_vec(const std::vector<T> &other) : hidl_vec() {
        *this = other;
    }

    template <typename InputIterator,
              typename = typename std::enable_if<std::is_convertible<
                  typename std::iterator_traits<InputIterator>::iterator_category,
                  std::input_iterator_tag>::value>::type>
    hidl_vec(InputIterator first, InputIterator last) : hidl_vec() {
        auto size = std::distance(first, last);
        if (size > static_cast<int64_t>(UINT32_MAX)) {
            details::logAlwaysFatal("hidl_vec can't hold more than 2^32 elements.");
        }
        if (size < 0) {
            details::logAlwaysFatal("size can't be negative.");
        }
        mSize = static_cast<uint32_t>(size);
        mBuffer = new T[mSize]();
        mOwnsBuffer = true;

        size_t idx = 0;
        for (; first != last; ++first) {
            mBuffer[idx++] = static_cast<T>(*first);
        }
    }

    ~hidl_vec() {
        if (mOwnsBuffer) {
            delete[] mBuffer;
        }
        mBuffer = nullptr;
    }

    // Reference an existing array, optionally taking ownership. It is the
    // caller's responsibility to ensure that the underlying memory stays
    // valid for the lifetime of this hidl_vec.
    void setToExternal(T *data, size_t size, bool shouldOwn = false) {
        if (mOwnsBuffer) {
            delete [] mBuffer;
        }
        mBuffer = data;
        if (size > UINT32_MAX) {
            details::logAlwaysFatal("external vector size exceeds 2^32 elements.");
        }
        mSize = static_cast<uint32_t>(size);
        mOwnsBuffer = shouldOwn;
    }

    T *data() {
        return mBuffer;
    }

    const T *data() const {
        return mBuffer;
    }

    T *releaseData() {
        if (!mOwnsBuffer && mSize > 0) {
            resize(mSize);
        }
        mOwnsBuffer = false;
        return mBuffer;
    }

    hidl_vec &operator=(hidl_vec &&other) noexcept {
        if (mOwnsBuffer) {
            delete[] mBuffer;
        }
        mBuffer = other.mBuffer;
        mSize = other.mSize;
        mOwnsBuffer = other.mOwnsBuffer;
        other.mOwnsBuffer = false;
        return *this;
    }

    hidl_vec &operator=(const hidl_vec &other) {
        if (this != &other) {
            if (mOwnsBuffer) {
                delete[] mBuffer;
            }
            copyFrom(other, other.mSize);
        }

        return *this;
    }

    // copy from an std::vector.
    hidl_vec &operator=(const std::vector<T> &other) {
        if (mOwnsBuffer) {
            delete[] mBuffer;
        }
        copyFrom(other, other.size());
        return *this;
    }

    // cast to an std::vector.
    operator std::vector<T>() const {
        std::vector<T> v(mSize);
        for (size_t i = 0; i < mSize; ++i) {
            v[i] = mBuffer[i];
        }
        return v;
    }

    // equality check, assuming that T::operator== is defined.
    bool operator==(const hidl_vec &other) const {
        if (mSize != other.size()) {
            return false;
        }
        for (size_t i = 0; i < mSize; ++i) {
            if (!(mBuffer[i] == other.mBuffer[i])) {
                return false;
            }
        }
        return true;
    }

    // inequality check, assuming that T::operator== is defined.
    inline bool operator!=(const hidl_vec &other) const {
        return !((*this) == other);
    }

    size_t size() const {
        return mSize;
    }

    T &operator[](size_t index) {
        return mBuffer[index];
    }

    const T &operator[](size_t index) const {
        return mBuffer[index];
    }

    // Does not initialize primitive types if new size > old size.
    void resize(size_t size) {
        if (size > UINT32_MAX) {
            details::logAlwaysFatal("hidl_vec can't hold more than 2^32 elements.");
        }
        T* newBuffer = new T[size]();

        for (size_t i = 0; i < std::min(static_cast<uint32_t>(size), mSize); ++i) {
            newBuffer[i] = mBuffer[i];
        }

        if (mOwnsBuffer) {
            delete[] mBuffer;
        }
        mBuffer = newBuffer;

        mSize = static_cast<uint32_t>(size);
        mOwnsBuffer = true;
    }

    // offsetof(hidl_string, mBuffer) exposed since mBuffer is private.
    static const size_t kOffsetOfBuffer;

private:
    // Define std interator interface for walking the array contents
    template<bool is_const>
    class iter : public std::iterator<
            std::random_access_iterator_tag, /* Category */
            T,
            ptrdiff_t, /* Distance */
            typename std::conditional<is_const, const T *, T *>::type /* Pointer */,
            typename std::conditional<is_const, const T &, T &>::type /* Reference */>
    {
        using traits = std::iterator_traits<iter>;
        using ptr_type = typename traits::pointer;
        using ref_type = typename traits::reference;
        using diff_type = typename traits::difference_type;
    public:
        iter(ptr_type ptr) : mPtr(ptr) { }
        inline iter &operator++()    { mPtr++; return *this; }
        inline iter  operator++(int) { iter i = *this; mPtr++; return i; }
        inline iter &operator--()    { mPtr--; return *this; }
        inline iter  operator--(int) { iter i = *this; mPtr--; return i; }
        inline friend iter operator+(diff_type n, const iter &it) { return it.mPtr + n; }
        inline iter  operator+(diff_type n) const { return mPtr + n; }
        inline iter  operator-(diff_type n) const { return mPtr - n; }
        inline diff_type operator-(const iter &other) const { return mPtr - other.mPtr; }
        inline iter &operator+=(diff_type n) { mPtr += n; return *this; }
        inline iter &operator-=(diff_type n) { mPtr -= n; return *this; }
        inline ref_type operator*() const  { return *mPtr; }
        inline ptr_type operator->() const { return mPtr; }
        inline bool operator==(const iter &rhs) const { return mPtr == rhs.mPtr; }
        inline bool operator!=(const iter &rhs) const { return mPtr != rhs.mPtr; }
        inline bool operator< (const iter &rhs) const { return mPtr <  rhs.mPtr; }
        inline bool operator> (const iter &rhs) const { return mPtr >  rhs.mPtr; }
        inline bool operator<=(const iter &rhs) const { return mPtr <= rhs.mPtr; }
        inline bool operator>=(const iter &rhs) const { return mPtr >= rhs.mPtr; }
        inline ref_type operator[](size_t n) const { return mPtr[n]; }
    private:
        ptr_type mPtr;
    };
public:
    using iterator       = iter<false /* is_const */>;
    using const_iterator = iter<true  /* is_const */>;

    iterator begin() { return data(); }
    iterator end() { return data()+mSize; }
    const_iterator begin() const { return data(); }
    const_iterator end() const { return data()+mSize; }

private:
    details::hidl_pointer<T> mBuffer;
    uint32_t mSize;
    bool mOwnsBuffer;

    // copy from an array-like object, assuming my resources are freed.
    template <typename Array>
    void copyFrom(const Array &data, size_t size) {
        mSize = static_cast<uint32_t>(size);
        mOwnsBuffer = true;
        if (mSize > 0) {
            mBuffer = new T[size]();
            for (size_t i = 0; i < size; ++i) {
                mBuffer[i] = data[i];
            }
        } else {
            mBuffer = nullptr;
        }
    }
};

template <typename T>
const size_t hidl_vec<T>::kOffsetOfBuffer = offsetof(hidl_vec<T>, mBuffer);

////////////////////////////////////////////////////////////////////////////////

namespace details {

    template<size_t SIZE1, size_t... SIZES>
    struct product {
        static constexpr size_t value = SIZE1 * product<SIZES...>::value;
    };

    template<size_t SIZE1>
    struct product<SIZE1> {
        static constexpr size_t value = SIZE1;
    };

    template<typename T, size_t SIZE1, size_t... SIZES>
    struct std_array {
        using type = std::array<typename std_array<T, SIZES...>::type, SIZE1>;
    };

    template<typename T, size_t SIZE1>
    struct std_array<T, SIZE1> {
        using type = std::array<T, SIZE1>;
    };

    template<typename T, size_t SIZE1, size_t... SIZES>
    struct accessor {

        using std_array_type = typename std_array<T, SIZE1, SIZES...>::type;

        explicit accessor(T *base)
            : mBase(base) {
        }

        accessor<T, SIZES...> operator[](size_t index) {
            return accessor<T, SIZES...>(
                    &mBase[index * product<SIZES...>::value]);
        }

        accessor &operator=(const std_array_type &other) {
            for (size_t i = 0; i < SIZE1; ++i) {
                (*this)[i] = other[i];
            }
            return *this;
        }

    private:
        T *mBase;
    };

    template<typename T, size_t SIZE1>
    struct accessor<T, SIZE1> {

        using std_array_type = typename std_array<T, SIZE1>::type;

        explicit accessor(T *base)
            : mBase(base) {
        }

        T &operator[](size_t index) {
            return mBase[index];
        }

        accessor &operator=(const std_array_type &other) {
            for (size_t i = 0; i < SIZE1; ++i) {
                (*this)[i] = other[i];
            }
            return *this;
        }

    private:
        T *mBase;
    };

    template<typename T, size_t SIZE1, size_t... SIZES>
    struct const_accessor {

        using std_array_type = typename std_array<T, SIZE1, SIZES...>::type;

        explicit const_accessor(const T *base)
            : mBase(base) {
        }

        const_accessor<T, SIZES...> operator[](size_t index) const {
            return const_accessor<T, SIZES...>(
                    &mBase[index * product<SIZES...>::value]);
        }

        operator std_array_type() {
            std_array_type array;
            for (size_t i = 0; i < SIZE1; ++i) {
                array[i] = (*this)[i];
            }
            return array;
        }

    private:
        const T *mBase;
    };

    template<typename T, size_t SIZE1>
    struct const_accessor<T, SIZE1> {

        using std_array_type = typename std_array<T, SIZE1>::type;

        explicit const_accessor(const T *base)
            : mBase(base) {
        }

        const T &operator[](size_t index) const {
            return mBase[index];
        }

        operator std_array_type() {
            std_array_type array;
            for (size_t i = 0; i < SIZE1; ++i) {
                array[i] = (*this)[i];
            }
            return array;
        }

    private:
        const T *mBase;
    };

}  // namespace details

////////////////////////////////////////////////////////////////////////////////

// A multidimensional array of T's. Assumes that T::operator=(const T &) is defined.
template<typename T, size_t SIZE1, size_t... SIZES>
struct hidl_array {

    using std_array_type = typename details::std_array<T, SIZE1, SIZES...>::type;

    hidl_array() = default;

    // Copies the data from source, using T::operator=(const T &).
    hidl_array(const T *source) {
        for (size_t i = 0; i < elementCount(); ++i) {
            mBuffer[i] = source[i];
        }
    }

    // Copies the data from the given std::array, using T::operator=(const T &).
    hidl_array(const std_array_type &array) {
        details::accessor<T, SIZE1, SIZES...> modifier(mBuffer);
        modifier = array;
    }

    T *data() { return mBuffer; }
    const T *data() const { return mBuffer; }

    details::accessor<T, SIZES...> operator[](size_t index) {
        return details::accessor<T, SIZES...>(
                &mBuffer[index * details::product<SIZES...>::value]);
    }

    details::const_accessor<T, SIZES...> operator[](size_t index) const {
        return details::const_accessor<T, SIZES...>(
                &mBuffer[index * details::product<SIZES...>::value]);
    }

    // equality check, assuming that T::operator== is defined.
    bool operator==(const hidl_array &other) const {
        for (size_t i = 0; i < elementCount(); ++i) {
            if (!(mBuffer[i] == other.mBuffer[i])) {
                return false;
            }
        }
        return true;
    }

    inline bool operator!=(const hidl_array &other) const {
        return !((*this) == other);
    }

    using size_tuple_type = std::tuple<decltype(SIZE1), decltype(SIZES)...>;

    static constexpr size_tuple_type size() {
        return std::make_tuple(SIZE1, SIZES...);
    }

    static constexpr size_t elementCount() {
        return details::product<SIZE1, SIZES...>::value;
    }

    operator std_array_type() const {
        return details::const_accessor<T, SIZE1, SIZES...>(mBuffer);
    }

private:
    T mBuffer[elementCount()];
};

// An array of T's. Assumes that T::operator=(const T &) is defined.
template<typename T, size_t SIZE1>
struct hidl_array<T, SIZE1> {

    using std_array_type = typename details::std_array<T, SIZE1>::type;

    hidl_array() = default;

    // Copies the data from source, using T::operator=(const T &).
    hidl_array(const T *source) {
        for (size_t i = 0; i < elementCount(); ++i) {
            mBuffer[i] = source[i];
        }
    }

    // Copies the data from the given std::array, using T::operator=(const T &).
    hidl_array(const std_array_type &array) : hidl_array(array.data()) {}

    T *data() { return mBuffer; }
    const T *data() const { return mBuffer; }

    T &operator[](size_t index) {
        return mBuffer[index];
    }

    const T &operator[](size_t index) const {
        return mBuffer[index];
    }

    // equality check, assuming that T::operator== is defined.
    bool operator==(const hidl_array &other) const {
        for (size_t i = 0; i < elementCount(); ++i) {
            if (!(mBuffer[i] == other.mBuffer[i])) {
                return false;
            }
        }
        return true;
    }

    inline bool operator!=(const hidl_array &other) const {
        return !((*this) == other);
    }

    static constexpr size_t size() { return SIZE1; }
    static constexpr size_t elementCount() { return SIZE1; }

    // Copies the data to an std::array, using T::operator=(T).
    operator std_array_type() const {
        std_array_type array;
        for (size_t i = 0; i < SIZE1; ++i) {
            array[i] = mBuffer[i];
        }
        return array;
    }

private:
    T mBuffer[SIZE1];
};

// ----------------------------------------------------------------------
// Version functions
struct hidl_version {
public:
    constexpr hidl_version(uint16_t major, uint16_t minor) : mMajor(major), mMinor(minor) {
        static_assert(sizeof(*this) == 4, "wrong size");
    }

    bool operator==(const hidl_version& other) const {
        return (mMajor == other.get_major() && mMinor == other.get_minor());
    }

    bool operator<(const hidl_version& other) const {
        return (mMajor < other.get_major() ||
                (mMajor == other.get_major() && mMinor < other.get_minor()));
    }

    bool operator>(const hidl_version& other) const {
        return other < *this;
    }

    bool operator<=(const hidl_version& other) const {
        return !(*this > other);
    }

    bool operator>=(const hidl_version& other) const {
        return !(*this < other);
    }

    constexpr uint16_t get_major() const { return mMajor; }
    constexpr uint16_t get_minor() const { return mMinor; }

private:
    uint16_t mMajor;
    uint16_t mMinor;
};

inline android::hardware::hidl_version make_hidl_version(uint16_t major, uint16_t minor) {
    return hidl_version(major,minor);
}

///////////////////// toString functions

std::string toString(const void *t);

// toString alias for numeric types
template<typename T, typename = typename std::enable_if<std::is_arithmetic<T>::value, T>::type>
inline std::string toString(T t) {
    return std::to_string(t);
}

namespace details {

template<typename T, typename = typename std::enable_if<std::is_arithmetic<T>::value, T>::type>
inline std::string toHexString(T t, bool prefix = true) {
    std::ostringstream os;
    if (prefix) { os << std::showbase; }
    os << std::hex << t;
    return os.str();
}

template<>
inline std::string toHexString(uint8_t t, bool prefix) {
    return toHexString(static_cast<int32_t>(t), prefix);
}

template<>
inline std::string toHexString(int8_t t, bool prefix) {
    return toHexString(static_cast<int32_t>(t), prefix);
}

template<typename Array>
std::string arrayToString(const Array &a, size_t size);

template<size_t SIZE1>
std::string arraySizeToString() {
    return std::string{"["} + toString(SIZE1) + "]";
}

template<size_t SIZE1, size_t SIZE2, size_t... SIZES>
std::string arraySizeToString() {
    return std::string{"["} + toString(SIZE1) + "]" + arraySizeToString<SIZE2, SIZES...>();
}

template<typename T, size_t SIZE1>
std::string toString(details::const_accessor<T, SIZE1> a) {
    return arrayToString(a, SIZE1);
}

template<typename Array>
std::string arrayToString(const Array &a, size_t size) {
    using android::hardware::toString;
    std::string os;
    os += "{";
    for (size_t i = 0; i < size; ++i) {
        if (i > 0) {
            os += ", ";
        }
        os += toString(a[i]);
    }
    os += "}";
    return os;
}

template<typename T, size_t SIZE1, size_t SIZE2, size_t... SIZES>
std::string toString(details::const_accessor<T, SIZE1, SIZE2, SIZES...> a) {
    return arrayToString(a, SIZE1);
}

}  //namespace details

inline std::string toString(const void *t) {
    return details::toHexString(reinterpret_cast<uintptr_t>(t));
}

// debug string dump. There will be quotes around the string!
inline std::string toString(const hidl_string &hs) {
    return std::string{"\""} + hs.c_str() + "\"";
}

// debug string dump
inline std::string toString(const hidl_handle &hs) {
    return toString(hs.getNativeHandle());
}

inline std::string toString(const hidl_memory &mem) {
    return std::string{"memory {.name = "} + toString(mem.name()) + ", .size = "
              + toString(mem.size())
              + ", .handle = " + toString(mem.handle()) + "}";
}

inline std::string toString(const sp<hidl_death_recipient> &dr) {
    return std::string{"death_recipient@"} + toString(dr.get());
}

// debug string dump, assuming that toString(T) is defined.
template<typename T>
std::string toString(const hidl_vec<T> &a) {
    std::string os;
    os += "[" + toString(a.size()) + "]";
    os += details::arrayToString(a, a.size());
    return os;
}

template<typename T, size_t SIZE1>
std::string toString(const hidl_array<T, SIZE1> &a) {
    return details::arraySizeToString<SIZE1>()
            + details::toString(details::const_accessor<T, SIZE1>(a.data()));
}

template<typename T, size_t SIZE1, size_t SIZE2, size_t... SIZES>
std::string toString(const hidl_array<T, SIZE1, SIZE2, SIZES...> &a) {
    return details::arraySizeToString<SIZE1, SIZE2, SIZES...>()
            + details::toString(details::const_accessor<T, SIZE1, SIZE2, SIZES...>(a.data()));
}

/**
 * Every HIDL generated enum generates an implementation of this function.
 * E.x.: for(const auto v : hidl_enum_iterator<Enum>) { ... }
 */
template <typename>
struct hidl_enum_iterator;

/**
 * Bitfields in HIDL are the underlying type of the enumeration.
 */
template <typename Enum>
using hidl_bitfield = typename std::underlying_type<Enum>::type;

}  // namespace hardware
}  // namespace android


#endif  // ANDROID_HIDL_SUPPORT_H
