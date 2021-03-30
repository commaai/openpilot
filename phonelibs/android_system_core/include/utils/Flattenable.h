/*
 * Copyright (C) 2010 The Android Open Source Project
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

#ifndef ANDROID_UTILS_FLATTENABLE_H
#define ANDROID_UTILS_FLATTENABLE_H


#include <stdint.h>
#include <sys/types.h>
#include <utils/Errors.h>
#include <utils/Debug.h>

namespace android {


class FlattenableUtils {
public:
    template<int N>
    static size_t align(size_t size) {
        COMPILE_TIME_ASSERT_FUNCTION_SCOPE( !(N & (N-1)) );
        return (size + (N-1)) & ~(N-1);
    }

    template<int N>
    static size_t align(void const*& buffer) {
        COMPILE_TIME_ASSERT_FUNCTION_SCOPE( !(N & (N-1)) );
        intptr_t b = intptr_t(buffer);
        buffer = (void*)((intptr_t(buffer) + (N-1)) & ~(N-1));
        return size_t(intptr_t(buffer) - b);
    }

    template<int N>
    static size_t align(void*& buffer) {
        return align<N>( const_cast<void const*&>(buffer) );
    }

    static void advance(void*& buffer, size_t& size, size_t offset) {
        buffer = reinterpret_cast<void*>( intptr_t(buffer) + offset );
        size -= offset;
    }

    static void advance(void const*& buffer, size_t& size, size_t offset) {
        buffer = reinterpret_cast<void const*>( intptr_t(buffer) + offset );
        size -= offset;
    }

    // write a POD structure
    template<typename T>
    static void write(void*& buffer, size_t& size, const T& value) {
        *static_cast<T*>(buffer) = value;
        advance(buffer, size, sizeof(T));
    }

    // read a POD structure
    template<typename T>
    static void read(void const*& buffer, size_t& size, T& value) {
        value = *static_cast<T const*>(buffer);
        advance(buffer, size, sizeof(T));
    }
};


/*
 * The Flattenable protocol allows an object to serialize itself out
 * to a byte-buffer and an array of file descriptors.
 * Flattenable objects must implement this protocol.
 */

template <typename T>
class Flattenable {
public:
    // size in bytes of the flattened object
    inline size_t getFlattenedSize() const;

    // number of file descriptors to flatten
    inline size_t getFdCount() const;

    // flattens the object into buffer.
    // size should be at least of getFlattenedSize()
    // file descriptors are written in the fds[] array but ownership is
    // not transfered (ie: they must be dupped by the caller of
    // flatten() if needed).
    inline status_t flatten(void*& buffer, size_t& size, int*& fds, size_t& count) const;

    // unflattens the object from buffer.
    // size should be equal to the value of getFlattenedSize() when the
    // object was flattened.
    // unflattened file descriptors are found in the fds[] array and
    // don't need to be dupped(). ie: the caller of unflatten doesn't
    // keep ownership. If a fd is not retained by unflatten() it must be
    // explicitly closed.
    inline status_t unflatten(void const*& buffer, size_t& size, int const*& fds, size_t& count);
};

template<typename T>
inline size_t Flattenable<T>::getFlattenedSize() const {
    return static_cast<T const*>(this)->T::getFlattenedSize();
}
template<typename T>
inline size_t Flattenable<T>::getFdCount() const {
    return static_cast<T const*>(this)->T::getFdCount();
}
template<typename T>
inline status_t Flattenable<T>::flatten(
        void*& buffer, size_t& size, int*& fds, size_t& count) const {
    return static_cast<T const*>(this)->T::flatten(buffer, size, fds, count);
}
template<typename T>
inline status_t Flattenable<T>::unflatten(
        void const*& buffer, size_t& size, int const*& fds, size_t& count) {
    return static_cast<T*>(this)->T::unflatten(buffer, size, fds, count);
}

/*
 * LightFlattenable is a protocol allowing object to serialize themselves out
 * to a byte-buffer. Because it doesn't handle file-descriptors,
 * LightFlattenable is usually more size efficient than Flattenable.
 * LightFlattenable objects must implement this protocol.
 */
template <typename T>
class LightFlattenable {
public:
    // returns whether this object always flatten into the same size.
    // for efficiency, this should always be inline.
    inline bool isFixedSize() const;

    // returns size in bytes of the flattened object. must be a constant.
    inline size_t getFlattenedSize() const;

    // flattens the object into buffer.
    inline status_t flatten(void* buffer, size_t size) const;

    // unflattens the object from buffer of given size.
    inline status_t unflatten(void const* buffer, size_t size);
};

template <typename T>
inline bool LightFlattenable<T>::isFixedSize() const {
    return static_cast<T const*>(this)->T::isFixedSize();
}
template <typename T>
inline size_t LightFlattenable<T>::getFlattenedSize() const {
    return static_cast<T const*>(this)->T::getFlattenedSize();
}
template <typename T>
inline status_t LightFlattenable<T>::flatten(void* buffer, size_t size) const {
    return static_cast<T const*>(this)->T::flatten(buffer, size);
}
template <typename T>
inline status_t LightFlattenable<T>::unflatten(void const* buffer, size_t size) {
    return static_cast<T*>(this)->T::unflatten(buffer, size);
}

/*
 * LightFlattenablePod is an implementation of the LightFlattenable protocol
 * for POD (plain-old-data) objects.
 * Simply derive from LightFlattenablePod<Foo> to make Foo flattenable; no
 * need to implement any methods; obviously Foo must be a POD structure.
 */
template <typename T>
class LightFlattenablePod : public LightFlattenable<T> {
public:
    inline bool isFixedSize() const {
        return true;
    }

    inline size_t getFlattenedSize() const {
        return sizeof(T);
    }
    inline status_t flatten(void* buffer, size_t size) const {
        if (size < sizeof(T)) return NO_MEMORY;
        *reinterpret_cast<T*>(buffer) = *static_cast<T const*>(this);
        return NO_ERROR;
    }
    inline status_t unflatten(void const* buffer, size_t) {
        *static_cast<T*>(this) = *reinterpret_cast<T const*>(buffer);
        return NO_ERROR;
    }
};


}; // namespace android


#endif /* ANDROID_UTILS_FLATTENABLE_H */
