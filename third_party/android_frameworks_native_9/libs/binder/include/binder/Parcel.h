/*
 * Copyright (C) 2005 The Android Open Source Project
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

#ifndef ANDROID_PARCEL_H
#define ANDROID_PARCEL_H

#include <string>
#include <vector>

#include <android-base/unique_fd.h>
#include <cutils/native_handle.h>
#include <utils/Errors.h>
#include <utils/RefBase.h>
#include <utils/String16.h>
#include <utils/Vector.h>
#include <utils/Flattenable.h>
#include <linux/android/binder.h>

#include <binder/IInterface.h>
#include <binder/Parcelable.h>
#include <binder/Map.h>

// ---------------------------------------------------------------------------
namespace android {

template <typename T> class Flattenable;
template <typename T> class LightFlattenable;
class IBinder;
class IPCThreadState;
class ProcessState;
class String8;
class TextOutput;

namespace binder {
class Value;
};

class Parcel {
    friend class IPCThreadState;
public:
    class ReadableBlob;
    class WritableBlob;

                        Parcel();
                        ~Parcel();
    
    const uint8_t*      data() const;
    size_t              dataSize() const;
    size_t              dataAvail() const;
    size_t              dataPosition() const;
    size_t              dataCapacity() const;

    status_t            setDataSize(size_t size);
    void                setDataPosition(size_t pos) const;
    status_t            setDataCapacity(size_t size);
    
    status_t            setData(const uint8_t* buffer, size_t len);

    status_t            appendFrom(const Parcel *parcel,
                                   size_t start, size_t len);

    int                 compareData(const Parcel& other);

    bool                allowFds() const;
    bool                pushAllowFds(bool allowFds);
    void                restoreAllowFds(bool lastValue);

    bool                hasFileDescriptors() const;

    // Writes the RPC header.
    status_t            writeInterfaceToken(const String16& interface);

    // Parses the RPC header, returning true if the interface name
    // in the header matches the expected interface from the caller.
    //
    // Additionally, enforceInterface does part of the work of
    // propagating the StrictMode policy mask, populating the current
    // IPCThreadState, which as an optimization may optionally be
    // passed in.
    bool                enforceInterface(const String16& interface,
                                         IPCThreadState* threadState = NULL) const;
    bool                checkInterface(IBinder*) const;

    void                freeData();

private:
    const binder_size_t* objects() const;

public:
    size_t              objectsCount() const;
    
    status_t            errorCheck() const;
    void                setError(status_t err);
    
    status_t            write(const void* data, size_t len);
    void*               writeInplace(size_t len);
    status_t            writeUnpadded(const void* data, size_t len);
    status_t            writeInt32(int32_t val);
    status_t            writeUint32(uint32_t val);
    status_t            writeInt64(int64_t val);
    status_t            writeUint64(uint64_t val);
    status_t            writeFloat(float val);
    status_t            writeDouble(double val);
    status_t            writeCString(const char* str);
    status_t            writeString8(const String8& str);
    status_t            writeString16(const String16& str);
    status_t            writeString16(const std::unique_ptr<String16>& str);
    status_t            writeString16(const char16_t* str, size_t len);
    status_t            writeStrongBinder(const sp<IBinder>& val);
    status_t            writeWeakBinder(const wp<IBinder>& val);
    status_t            writeInt32Array(size_t len, const int32_t *val);
    status_t            writeByteArray(size_t len, const uint8_t *val);
    status_t            writeBool(bool val);
    status_t            writeChar(char16_t val);
    status_t            writeByte(int8_t val);

    // Take a UTF8 encoded string, convert to UTF16, write it to the parcel.
    status_t            writeUtf8AsUtf16(const std::string& str);
    status_t            writeUtf8AsUtf16(const std::unique_ptr<std::string>& str);

    status_t            writeByteVector(const std::unique_ptr<std::vector<int8_t>>& val);
    status_t            writeByteVector(const std::vector<int8_t>& val);
    status_t            writeByteVector(const std::unique_ptr<std::vector<uint8_t>>& val);
    status_t            writeByteVector(const std::vector<uint8_t>& val);
    status_t            writeInt32Vector(const std::unique_ptr<std::vector<int32_t>>& val);
    status_t            writeInt32Vector(const std::vector<int32_t>& val);
    status_t            writeInt64Vector(const std::unique_ptr<std::vector<int64_t>>& val);
    status_t            writeInt64Vector(const std::vector<int64_t>& val);
    status_t            writeFloatVector(const std::unique_ptr<std::vector<float>>& val);
    status_t            writeFloatVector(const std::vector<float>& val);
    status_t            writeDoubleVector(const std::unique_ptr<std::vector<double>>& val);
    status_t            writeDoubleVector(const std::vector<double>& val);
    status_t            writeBoolVector(const std::unique_ptr<std::vector<bool>>& val);
    status_t            writeBoolVector(const std::vector<bool>& val);
    status_t            writeCharVector(const std::unique_ptr<std::vector<char16_t>>& val);
    status_t            writeCharVector(const std::vector<char16_t>& val);
    status_t            writeString16Vector(
                            const std::unique_ptr<std::vector<std::unique_ptr<String16>>>& val);
    status_t            writeString16Vector(const std::vector<String16>& val);
    status_t            writeUtf8VectorAsUtf16Vector(
                            const std::unique_ptr<std::vector<std::unique_ptr<std::string>>>& val);
    status_t            writeUtf8VectorAsUtf16Vector(const std::vector<std::string>& val);

    status_t            writeStrongBinderVector(const std::unique_ptr<std::vector<sp<IBinder>>>& val);
    status_t            writeStrongBinderVector(const std::vector<sp<IBinder>>& val);

    template<typename T>
    status_t            writeParcelableVector(const std::unique_ptr<std::vector<std::unique_ptr<T>>>& val);
    template<typename T>
    status_t            writeParcelableVector(const std::shared_ptr<std::vector<std::unique_ptr<T>>>& val);
    template<typename T>
    status_t            writeParcelableVector(const std::vector<T>& val);

    template<typename T>
    status_t            writeNullableParcelable(const std::unique_ptr<T>& parcelable);

    status_t            writeParcelable(const Parcelable& parcelable);

    status_t            writeValue(const binder::Value& value);

    template<typename T>
    status_t            write(const Flattenable<T>& val);

    template<typename T>
    status_t            write(const LightFlattenable<T>& val);

    template<typename T>
    status_t            writeVectorSize(const std::vector<T>& val);
    template<typename T>
    status_t            writeVectorSize(const std::unique_ptr<std::vector<T>>& val);

    status_t            writeMap(const binder::Map& map);
    status_t            writeNullableMap(const std::unique_ptr<binder::Map>& map);

    // Place a native_handle into the parcel (the native_handle's file-
    // descriptors are dup'ed, so it is safe to delete the native_handle
    // when this function returns).
    // Doesn't take ownership of the native_handle.
    status_t            writeNativeHandle(const native_handle* handle);

    // Place a file descriptor into the parcel.  The given fd must remain
    // valid for the lifetime of the parcel.
    // The Parcel does not take ownership of the given fd unless you ask it to.
    status_t            writeFileDescriptor(int fd, bool takeOwnership = false);

    // Place a file descriptor into the parcel.  A dup of the fd is made, which
    // will be closed once the parcel is destroyed.
    status_t            writeDupFileDescriptor(int fd);

    // Place a Java "parcel file descriptor" into the parcel.  The given fd must remain
    // valid for the lifetime of the parcel.
    // The Parcel does not take ownership of the given fd unless you ask it to.
    status_t            writeParcelFileDescriptor(int fd, bool takeOwnership = false);

    // Place a file descriptor into the parcel.  This will not affect the
    // semantics of the smart file descriptor. A new descriptor will be
    // created, and will be closed when the parcel is destroyed.
    status_t            writeUniqueFileDescriptor(
                            const base::unique_fd& fd);

    // Place a vector of file desciptors into the parcel. Each descriptor is
    // dup'd as in writeDupFileDescriptor
    status_t            writeUniqueFileDescriptorVector(
                            const std::unique_ptr<std::vector<base::unique_fd>>& val);
    status_t            writeUniqueFileDescriptorVector(
                            const std::vector<base::unique_fd>& val);

    // Writes a blob to the parcel.
    // If the blob is small, then it is stored in-place, otherwise it is
    // transferred by way of an anonymous shared memory region.  Prefer sending
    // immutable blobs if possible since they may be subsequently transferred between
    // processes without further copying whereas mutable blobs always need to be copied.
    // The caller should call release() on the blob after writing its contents.
    status_t            writeBlob(size_t len, bool mutableCopy, WritableBlob* outBlob);

    // Write an existing immutable blob file descriptor to the parcel.
    // This allows the client to send the same blob to multiple processes
    // as long as it keeps a dup of the blob file descriptor handy for later.
    status_t            writeDupImmutableBlobFileDescriptor(int fd);

    status_t            writeObject(const flat_binder_object& val, bool nullMetaData);

    // Like Parcel.java's writeNoException().  Just writes a zero int32.
    // Currently the native implementation doesn't do any of the StrictMode
    // stack gathering and serialization that the Java implementation does.
    status_t            writeNoException();

    void                remove(size_t start, size_t amt);
    
    status_t            read(void* outData, size_t len) const;
    const void*         readInplace(size_t len) const;
    int32_t             readInt32() const;
    status_t            readInt32(int32_t *pArg) const;
    uint32_t            readUint32() const;
    status_t            readUint32(uint32_t *pArg) const;
    int64_t             readInt64() const;
    status_t            readInt64(int64_t *pArg) const;
    uint64_t            readUint64() const;
    status_t            readUint64(uint64_t *pArg) const;
    float               readFloat() const;
    status_t            readFloat(float *pArg) const;
    double              readDouble() const;
    status_t            readDouble(double *pArg) const;
    intptr_t            readIntPtr() const;
    status_t            readIntPtr(intptr_t *pArg) const;
    bool                readBool() const;
    status_t            readBool(bool *pArg) const;
    char16_t            readChar() const;
    status_t            readChar(char16_t *pArg) const;
    int8_t              readByte() const;
    status_t            readByte(int8_t *pArg) const;

    // Read a UTF16 encoded string, convert to UTF8
    status_t            readUtf8FromUtf16(std::string* str) const;
    status_t            readUtf8FromUtf16(std::unique_ptr<std::string>* str) const;

    const char*         readCString() const;
    String8             readString8() const;
    status_t            readString8(String8* pArg) const;
    String16            readString16() const;
    status_t            readString16(String16* pArg) const;
    status_t            readString16(std::unique_ptr<String16>* pArg) const;
    const char16_t*     readString16Inplace(size_t* outLen) const;
    sp<IBinder>         readStrongBinder() const;
    status_t            readStrongBinder(sp<IBinder>* val) const;
    status_t            readNullableStrongBinder(sp<IBinder>* val) const;
    wp<IBinder>         readWeakBinder() const;

    template<typename T>
    status_t            readParcelableVector(
                            std::unique_ptr<std::vector<std::unique_ptr<T>>>* val) const;
    template<typename T>
    status_t            readParcelableVector(std::vector<T>* val) const;

    status_t            readParcelable(Parcelable* parcelable) const;

    template<typename T>
    status_t            readParcelable(std::unique_ptr<T>* parcelable) const;

    status_t            readValue(binder::Value* value) const;

    template<typename T>
    status_t            readStrongBinder(sp<T>* val) const;

    template<typename T>
    status_t            readNullableStrongBinder(sp<T>* val) const;

    status_t            readStrongBinderVector(std::unique_ptr<std::vector<sp<IBinder>>>* val) const;
    status_t            readStrongBinderVector(std::vector<sp<IBinder>>* val) const;

    status_t            readByteVector(std::unique_ptr<std::vector<int8_t>>* val) const;
    status_t            readByteVector(std::vector<int8_t>* val) const;
    status_t            readByteVector(std::unique_ptr<std::vector<uint8_t>>* val) const;
    status_t            readByteVector(std::vector<uint8_t>* val) const;
    status_t            readInt32Vector(std::unique_ptr<std::vector<int32_t>>* val) const;
    status_t            readInt32Vector(std::vector<int32_t>* val) const;
    status_t            readInt64Vector(std::unique_ptr<std::vector<int64_t>>* val) const;
    status_t            readInt64Vector(std::vector<int64_t>* val) const;
    status_t            readFloatVector(std::unique_ptr<std::vector<float>>* val) const;
    status_t            readFloatVector(std::vector<float>* val) const;
    status_t            readDoubleVector(std::unique_ptr<std::vector<double>>* val) const;
    status_t            readDoubleVector(std::vector<double>* val) const;
    status_t            readBoolVector(std::unique_ptr<std::vector<bool>>* val) const;
    status_t            readBoolVector(std::vector<bool>* val) const;
    status_t            readCharVector(std::unique_ptr<std::vector<char16_t>>* val) const;
    status_t            readCharVector(std::vector<char16_t>* val) const;
    status_t            readString16Vector(
                            std::unique_ptr<std::vector<std::unique_ptr<String16>>>* val) const;
    status_t            readString16Vector(std::vector<String16>* val) const;
    status_t            readUtf8VectorFromUtf16Vector(
                            std::unique_ptr<std::vector<std::unique_ptr<std::string>>>* val) const;
    status_t            readUtf8VectorFromUtf16Vector(std::vector<std::string>* val) const;

    template<typename T>
    status_t            read(Flattenable<T>& val) const;

    template<typename T>
    status_t            read(LightFlattenable<T>& val) const;

    template<typename T>
    status_t            resizeOutVector(std::vector<T>* val) const;
    template<typename T>
    status_t            resizeOutVector(std::unique_ptr<std::vector<T>>* val) const;

    status_t            readMap(binder::Map* map)const;
    status_t            readNullableMap(std::unique_ptr<binder::Map>* map) const;

    // Like Parcel.java's readExceptionCode().  Reads the first int32
    // off of a Parcel's header, returning 0 or the negative error
    // code on exceptions, but also deals with skipping over rich
    // response headers.  Callers should use this to read & parse the
    // response headers rather than doing it by hand.
    int32_t             readExceptionCode() const;

    // Retrieve native_handle from the parcel. This returns a copy of the
    // parcel's native_handle (the caller takes ownership). The caller
    // must free the native_handle with native_handle_close() and 
    // native_handle_delete().
    native_handle*     readNativeHandle() const;

    
    // Retrieve a file descriptor from the parcel.  This returns the raw fd
    // in the parcel, which you do not own -- use dup() to get your own copy.
    int                 readFileDescriptor() const;

    // Retrieve a Java "parcel file descriptor" from the parcel.  This returns the raw fd
    // in the parcel, which you do not own -- use dup() to get your own copy.
    int                 readParcelFileDescriptor() const;

    // Retrieve a smart file descriptor from the parcel.
    status_t            readUniqueFileDescriptor(
                            base::unique_fd* val) const;


    // Retrieve a vector of smart file descriptors from the parcel.
    status_t            readUniqueFileDescriptorVector(
                            std::unique_ptr<std::vector<base::unique_fd>>* val) const;
    status_t            readUniqueFileDescriptorVector(
                            std::vector<base::unique_fd>* val) const;

    // Reads a blob from the parcel.
    // The caller should call release() on the blob after reading its contents.
    status_t            readBlob(size_t len, ReadableBlob* outBlob) const;

    const flat_binder_object* readObject(bool nullMetaData) const;

    // Explicitly close all file descriptors in the parcel.
    void                closeFileDescriptors();

    // Debugging: get metrics on current allocations.
    static size_t       getGlobalAllocSize();
    static size_t       getGlobalAllocCount();

private:
    typedef void        (*release_func)(Parcel* parcel,
                                        const uint8_t* data, size_t dataSize,
                                        const binder_size_t* objects, size_t objectsSize,
                                        void* cookie);
                        
    uintptr_t           ipcData() const;
    size_t              ipcDataSize() const;
    uintptr_t           ipcObjects() const;
    size_t              ipcObjectsCount() const;
    void                ipcSetDataReference(const uint8_t* data, size_t dataSize,
                                            const binder_size_t* objects, size_t objectsCount,
                                            release_func relFunc, void* relCookie);
    
public:
    void                print(TextOutput& to, uint32_t flags = 0) const;

private:
                        Parcel(const Parcel& o);
    Parcel&             operator=(const Parcel& o);
    
    status_t            finishWrite(size_t len);
    void                releaseObjects();
    void                acquireObjects();
    status_t            growData(size_t len);
    status_t            restartWrite(size_t desired);
    status_t            continueWrite(size_t desired);
    status_t            writePointer(uintptr_t val);
    status_t            readPointer(uintptr_t *pArg) const;
    uintptr_t           readPointer() const;
    void                freeDataNoInit();
    void                initState();
    void                scanForFds() const;
    status_t            validateReadData(size_t len) const;
                        
    template<class T>
    status_t            readAligned(T *pArg) const;

    template<class T>   T readAligned() const;

    template<class T>
    status_t            writeAligned(T val);

    status_t            writeRawNullableParcelable(const Parcelable*
                                                   parcelable);

    template<typename T, typename U>
    status_t            unsafeReadTypedVector(std::vector<T>* val,
                                              status_t(Parcel::*read_func)(U*) const) const;
    template<typename T>
    status_t            readNullableTypedVector(std::unique_ptr<std::vector<T>>* val,
                                                status_t(Parcel::*read_func)(T*) const) const;
    template<typename T>
    status_t            readTypedVector(std::vector<T>* val,
                                        status_t(Parcel::*read_func)(T*) const) const;
    template<typename T, typename U>
    status_t            unsafeWriteTypedVector(const std::vector<T>& val,
                                               status_t(Parcel::*write_func)(U));
    template<typename T>
    status_t            writeNullableTypedVector(const std::unique_ptr<std::vector<T>>& val,
                                                 status_t(Parcel::*write_func)(const T&));
    template<typename T>
    status_t            writeNullableTypedVector(const std::unique_ptr<std::vector<T>>& val,
                                                 status_t(Parcel::*write_func)(T));
    template<typename T>
    status_t            writeTypedVector(const std::vector<T>& val,
                                         status_t(Parcel::*write_func)(const T&));
    template<typename T>
    status_t            writeTypedVector(const std::vector<T>& val,
                                         status_t(Parcel::*write_func)(T));

    status_t            mError;
    uint8_t*            mData;
    size_t              mDataSize;
    size_t              mDataCapacity;
    mutable size_t      mDataPos;
    binder_size_t*      mObjects;
    size_t              mObjectsSize;
    size_t              mObjectsCapacity;
    mutable size_t      mNextObjectHint;
    mutable bool        mObjectsSorted;

    mutable bool        mFdsKnown;
    mutable bool        mHasFds;
    bool                mAllowFds;

    release_func        mOwner;
    void*               mOwnerCookie;

    class Blob {
    public:
        Blob();
        ~Blob();

        void clear();
        void release();
        inline size_t size() const { return mSize; }
        inline int fd() const { return mFd; }
        inline bool isMutable() const { return mMutable; }

    protected:
        void init(int fd, void* data, size_t size, bool isMutable);

        int mFd; // owned by parcel so not closed when released
        void* mData;
        size_t mSize;
        bool mMutable;
    };

    #if defined(__clang__)
    #pragma clang diagnostic push
    #pragma clang diagnostic ignored "-Wweak-vtables"
    #endif

    // FlattenableHelperInterface and FlattenableHelper avoid generating a vtable entry in objects
    // following Flattenable template/protocol.
    class FlattenableHelperInterface {
    protected:
        ~FlattenableHelperInterface() { }
    public:
        virtual size_t getFlattenedSize() const = 0;
        virtual size_t getFdCount() const = 0;
        virtual status_t flatten(void* buffer, size_t size, int* fds, size_t count) const = 0;
        virtual status_t unflatten(void const* buffer, size_t size, int const* fds, size_t count) = 0;
    };

    #if defined(__clang__)
    #pragma clang diagnostic pop
    #endif

    // Concrete implementation of FlattenableHelperInterface that delegates virtual calls to the
    // specified class T implementing the Flattenable protocol. It "virtualizes" a compile-time
    // protocol.
    template<typename T>
    class FlattenableHelper : public FlattenableHelperInterface {
        friend class Parcel;
        const Flattenable<T>& val;
        explicit FlattenableHelper(const Flattenable<T>& _val) : val(_val) { }

    protected:
        ~FlattenableHelper() = default;
    public:
        virtual size_t getFlattenedSize() const {
            return val.getFlattenedSize();
        }
        virtual size_t getFdCount() const {
            return val.getFdCount();
        }
        virtual status_t flatten(void* buffer, size_t size, int* fds, size_t count) const {
            return val.flatten(buffer, size, fds, count);
        }
        virtual status_t unflatten(void const* buffer, size_t size, int const* fds, size_t count) {
            return const_cast<Flattenable<T>&>(val).unflatten(buffer, size, fds, count);
        }
    };
    status_t write(const FlattenableHelperInterface& val);
    status_t read(FlattenableHelperInterface& val) const;

public:
    class ReadableBlob : public Blob {
        friend class Parcel;
    public:
        inline const void* data() const { return mData; }
        inline void* mutableData() { return isMutable() ? mData : NULL; }
    };

    class WritableBlob : public Blob {
        friend class Parcel;
    public:
        inline void* data() { return mData; }
    };

private:
    size_t mOpenAshmemSize;

public:
    // TODO: Remove once ABI can be changed.
    size_t getBlobAshmemSize() const;
    size_t getOpenAshmemSize() const;
};

// ---------------------------------------------------------------------------

template<typename T>
status_t Parcel::write(const Flattenable<T>& val) {
    const FlattenableHelper<T> helper(val);
    return write(helper);
}

template<typename T>
status_t Parcel::write(const LightFlattenable<T>& val) {
    size_t size(val.getFlattenedSize());
    if (!val.isFixedSize()) {
        if (size > INT32_MAX) {
            return BAD_VALUE;
        }
        status_t err = writeInt32(static_cast<int32_t>(size));
        if (err != NO_ERROR) {
            return err;
        }
    }
    if (size) {
        void* buffer = writeInplace(size);
        if (buffer == NULL)
            return NO_MEMORY;
        return val.flatten(buffer, size);
    }
    return NO_ERROR;
}

template<typename T>
status_t Parcel::read(Flattenable<T>& val) const {
    FlattenableHelper<T> helper(val);
    return read(helper);
}

template<typename T>
status_t Parcel::read(LightFlattenable<T>& val) const {
    size_t size;
    if (val.isFixedSize()) {
        size = val.getFlattenedSize();
    } else {
        int32_t s;
        status_t err = readInt32(&s);
        if (err != NO_ERROR) {
            return err;
        }
        size = static_cast<size_t>(s);
    }
    if (size) {
        void const* buffer = readInplace(size);
        return buffer == NULL ? NO_MEMORY :
                val.unflatten(buffer, size);
    }
    return NO_ERROR;
}

template<typename T>
status_t Parcel::writeVectorSize(const std::vector<T>& val) {
    if (val.size() > INT32_MAX) {
        return BAD_VALUE;
    }
    return writeInt32(static_cast<int32_t>(val.size()));
}

template<typename T>
status_t Parcel::writeVectorSize(const std::unique_ptr<std::vector<T>>& val) {
    if (!val) {
        return writeInt32(-1);
    }

    return writeVectorSize(*val);
}

template<typename T>
status_t Parcel::resizeOutVector(std::vector<T>* val) const {
    int32_t size;
    status_t err = readInt32(&size);
    if (err != NO_ERROR) {
        return err;
    }

    if (size < 0) {
        return UNEXPECTED_NULL;
    }
    val->resize(size_t(size));
    return OK;
}

template<typename T>
status_t Parcel::resizeOutVector(std::unique_ptr<std::vector<T>>* val) const {
    int32_t size;
    status_t err = readInt32(&size);
    if (err != NO_ERROR) {
        return err;
    }

    val->reset();
    if (size >= 0) {
        val->reset(new std::vector<T>(size_t(size)));
    }

    return OK;
}

template<typename T>
status_t Parcel::readStrongBinder(sp<T>* val) const {
    sp<IBinder> tmp;
    status_t ret = readStrongBinder(&tmp);

    if (ret == OK) {
        *val = interface_cast<T>(tmp);

        if (val->get() == nullptr) {
            return UNKNOWN_ERROR;
        }
    }

    return ret;
}

template<typename T>
status_t Parcel::readNullableStrongBinder(sp<T>* val) const {
    sp<IBinder> tmp;
    status_t ret = readNullableStrongBinder(&tmp);

    if (ret == OK) {
        *val = interface_cast<T>(tmp);

        if (val->get() == nullptr && tmp.get() != nullptr) {
            ret = UNKNOWN_ERROR;
        }
    }

    return ret;
}

template<typename T, typename U>
status_t Parcel::unsafeReadTypedVector(
        std::vector<T>* val,
        status_t(Parcel::*read_func)(U*) const) const {
    int32_t size;
    status_t status = this->readInt32(&size);

    if (status != OK) {
        return status;
    }

    if (size < 0) {
        return UNEXPECTED_NULL;
    }

    if (val->max_size() < static_cast<size_t>(size)) {
        return NO_MEMORY;
    }

    val->resize(static_cast<size_t>(size));

    if (val->size() < static_cast<size_t>(size)) {
        return NO_MEMORY;
    }

    for (auto& v: *val) {
        status = (this->*read_func)(&v);

        if (status != OK) {
            return status;
        }
    }

    return OK;
}

template<typename T>
status_t Parcel::readTypedVector(std::vector<T>* val,
                                 status_t(Parcel::*read_func)(T*) const) const {
    return unsafeReadTypedVector(val, read_func);
}

template<typename T>
status_t Parcel::readNullableTypedVector(std::unique_ptr<std::vector<T>>* val,
                                         status_t(Parcel::*read_func)(T*) const) const {
    const size_t start = dataPosition();
    int32_t size;
    status_t status = readInt32(&size);
    val->reset();

    if (status != OK || size < 0) {
        return status;
    }

    setDataPosition(start);
    val->reset(new std::vector<T>());

    status = unsafeReadTypedVector(val->get(), read_func);

    if (status != OK) {
        val->reset();
    }

    return status;
}

template<typename T, typename U>
status_t Parcel::unsafeWriteTypedVector(const std::vector<T>& val,
                                        status_t(Parcel::*write_func)(U)) {
    if (val.size() > std::numeric_limits<int32_t>::max()) {
        return BAD_VALUE;
    }

    status_t status = this->writeInt32(static_cast<int32_t>(val.size()));

    if (status != OK) {
        return status;
    }

    for (const auto& item : val) {
        status = (this->*write_func)(item);

        if (status != OK) {
            return status;
        }
    }

    return OK;
}

template<typename T>
status_t Parcel::writeTypedVector(const std::vector<T>& val,
                                  status_t(Parcel::*write_func)(const T&)) {
    return unsafeWriteTypedVector(val, write_func);
}

template<typename T>
status_t Parcel::writeTypedVector(const std::vector<T>& val,
                                  status_t(Parcel::*write_func)(T)) {
    return unsafeWriteTypedVector(val, write_func);
}

template<typename T>
status_t Parcel::writeNullableTypedVector(const std::unique_ptr<std::vector<T>>& val,
                                          status_t(Parcel::*write_func)(const T&)) {
    if (val.get() == nullptr) {
        return this->writeInt32(-1);
    }

    return unsafeWriteTypedVector(*val, write_func);
}

template<typename T>
status_t Parcel::writeNullableTypedVector(const std::unique_ptr<std::vector<T>>& val,
                                          status_t(Parcel::*write_func)(T)) {
    if (val.get() == nullptr) {
        return this->writeInt32(-1);
    }

    return unsafeWriteTypedVector(*val, write_func);
}

template<typename T>
status_t Parcel::readParcelableVector(std::vector<T>* val) const {
    return unsafeReadTypedVector<T, Parcelable>(val, &Parcel::readParcelable);
}

template<typename T>
status_t Parcel::readParcelableVector(std::unique_ptr<std::vector<std::unique_ptr<T>>>* val) const {
    const size_t start = dataPosition();
    int32_t size;
    status_t status = readInt32(&size);
    val->reset();

    if (status != OK || size < 0) {
        return status;
    }

    setDataPosition(start);
    val->reset(new std::vector<std::unique_ptr<T>>());

    status = unsafeReadTypedVector(val->get(), &Parcel::readParcelable<T>);

    if (status != OK) {
        val->reset();
    }

    return status;
}

template<typename T>
status_t Parcel::readParcelable(std::unique_ptr<T>* parcelable) const {
    const size_t start = dataPosition();
    int32_t present;
    status_t status = readInt32(&present);
    parcelable->reset();

    if (status != OK || !present) {
        return status;
    }

    setDataPosition(start);
    parcelable->reset(new T());

    status = readParcelable(parcelable->get());

    if (status != OK) {
        parcelable->reset();
    }

    return status;
}

template<typename T>
status_t Parcel::writeNullableParcelable(const std::unique_ptr<T>& parcelable) {
    return writeRawNullableParcelable(parcelable.get());
}

template<typename T>
status_t Parcel::writeParcelableVector(const std::vector<T>& val) {
    return unsafeWriteTypedVector<T,const Parcelable&>(val, &Parcel::writeParcelable);
}

template<typename T>
status_t Parcel::writeParcelableVector(const std::unique_ptr<std::vector<std::unique_ptr<T>>>& val) {
    if (val.get() == nullptr) {
        return this->writeInt32(-1);
    }

    return unsafeWriteTypedVector(*val, &Parcel::writeNullableParcelable<T>);
}

template<typename T>
status_t Parcel::writeParcelableVector(const std::shared_ptr<std::vector<std::unique_ptr<T>>>& val) {
    if (val.get() == nullptr) {
        return this->writeInt32(-1);
    }

    return unsafeWriteTypedVector(*val, &Parcel::writeNullableParcelable<T>);
}

// ---------------------------------------------------------------------------

inline TextOutput& operator<<(TextOutput& to, const Parcel& parcel)
{
    parcel.print(to);
    return to;
}

// ---------------------------------------------------------------------------

// Generic acquire and release of objects.
void acquire_object(const sp<ProcessState>& proc,
                    const flat_binder_object& obj, const void* who);
void release_object(const sp<ProcessState>& proc,
                    const flat_binder_object& obj, const void* who);

void flatten_binder(const sp<ProcessState>& proc,
                    const sp<IBinder>& binder, flat_binder_object* out);
void flatten_binder(const sp<ProcessState>& proc,
                    const wp<IBinder>& binder, flat_binder_object* out);
status_t unflatten_binder(const sp<ProcessState>& proc,
                          const flat_binder_object& flat, sp<IBinder>* out);
status_t unflatten_binder(const sp<ProcessState>& proc,
                          const flat_binder_object& flat, wp<IBinder>* out);

}; // namespace android

// ---------------------------------------------------------------------------

#endif // ANDROID_PARCEL_H
