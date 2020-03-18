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

#include <cutils/native_handle.h>
#include <utils/Errors.h>
#include <utils/RefBase.h>
#include <utils/String16.h>
#include <utils/Vector.h>
#include <utils/Flattenable.h>
#include <linux/binder.h>

// ---------------------------------------------------------------------------
namespace android {

template <typename T> class Flattenable;
template <typename T> class LightFlattenable;
class IBinder;
class IPCThreadState;
class ProcessState;
class String8;
class TextOutput;

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
    status_t            writeString16(const char16_t* str, size_t len);
    status_t            writeStrongBinder(const sp<IBinder>& val);
    status_t            writeWeakBinder(const wp<IBinder>& val);
    status_t            writeInt32Array(size_t len, const int32_t *val);
    status_t            writeByteArray(size_t len, const uint8_t *val);

    template<typename T>
    status_t            write(const Flattenable<T>& val);

    template<typename T>
    status_t            write(const LightFlattenable<T>& val);


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

    const char*         readCString() const;
    String8             readString8() const;
    String16            readString16() const;
    const char16_t*     readString16Inplace(size_t* outLen) const;
    sp<IBinder>         readStrongBinder() const;
    wp<IBinder>         readWeakBinder() const;

    template<typename T>
    status_t            read(Flattenable<T>& val) const;

    template<typename T>
    status_t            read(LightFlattenable<T>& val) const;

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
                        
    template<class T>
    status_t            readAligned(T *pArg) const;

    template<class T>   T readAligned() const;

    template<class T>
    status_t            writeAligned(T val);

    status_t            mError;
    uint8_t*            mData;
    size_t              mDataSize;
    size_t              mDataCapacity;
    mutable size_t      mDataPos;
    binder_size_t*      mObjects;
    size_t              mObjectsSize;
    size_t              mObjectsCapacity;
    mutable size_t      mNextObjectHint;

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
        inline int fd() const { return mFd; };
        inline bool isMutable() const { return mMutable; }

    protected:
        void init(int fd, void* data, size_t size, bool isMutable);

        int mFd; // owned by parcel so not closed when released
        void* mData;
        size_t mSize;
        bool mMutable;
    };

    class FlattenableHelperInterface {
    protected:
        ~FlattenableHelperInterface() { }
    public:
        virtual size_t getFlattenedSize() const = 0;
        virtual size_t getFdCount() const = 0;
        virtual status_t flatten(void* buffer, size_t size, int* fds, size_t count) const = 0;
        virtual status_t unflatten(void const* buffer, size_t size, int const* fds, size_t count) = 0;
    };

    template<typename T>
    class FlattenableHelper : public FlattenableHelperInterface {
        friend class Parcel;
        const Flattenable<T>& val;
        explicit FlattenableHelper(const Flattenable<T>& val) : val(val) { }

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

#ifndef DISABLE_ASHMEM_TRACKING
private:
    size_t mOpenAshmemSize;
#endif

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
        status_t err = writeInt32(size);
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
        size = s;
    }
    if (size) {
        void const* buffer = readInplace(size);
        return buffer == NULL ? NO_MEMORY :
                val.unflatten(buffer, size);
    }
    return NO_ERROR;
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
