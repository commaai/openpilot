/*
 * Copyright (C) 2007 The Android Open Source Project
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

#ifndef ANDROID_IMEMORY_H
#define ANDROID_IMEMORY_H

#include <stdint.h>
#include <sys/types.h>
#include <sys/mman.h>

#include <utils/RefBase.h>
#include <utils/Errors.h>
#include <binder/IInterface.h>

namespace android {

// ----------------------------------------------------------------------------

class IMemoryHeap : public IInterface
{
public:
    DECLARE_META_INTERFACE(MemoryHeap);

    // flags returned by getFlags()
    enum {
        READ_ONLY   = 0x00000001,
#ifdef USE_MEMORY_HEAP_ION
        USE_ION_FD  = 0x00008000
#else
        USE_ION_FD  = 0x00000008
#endif
    };

    virtual int         getHeapID() const = 0;
    virtual void*       getBase() const = 0;
    virtual size_t      getSize() const = 0;
    virtual uint32_t    getFlags() const = 0;
    virtual uint32_t    getOffset() const = 0;

    // these are there just for backward source compatibility
    int32_t heapID() const { return getHeapID(); }
    void*   base() const  { return getBase(); }
    size_t  virtualSize() const { return getSize(); }
};

class BnMemoryHeap : public BnInterface<IMemoryHeap>
{
public:
    virtual status_t onTransact( 
            uint32_t code,
            const Parcel& data,
            Parcel* reply,
            uint32_t flags = 0);
    
    BnMemoryHeap();
protected:
    virtual ~BnMemoryHeap();
};

// ----------------------------------------------------------------------------

class IMemory : public IInterface
{
public:
    DECLARE_META_INTERFACE(Memory);

    virtual sp<IMemoryHeap> getMemory(ssize_t* offset=0, size_t* size=0) const = 0;

    // helpers
    void* fastPointer(const sp<IBinder>& heap, ssize_t offset) const;
    void* pointer() const;
    size_t size() const;
    ssize_t offset() const;
};

class BnMemory : public BnInterface<IMemory>
{
public:
    virtual status_t onTransact(
            uint32_t code,
            const Parcel& data,
            Parcel* reply,
            uint32_t flags = 0);

    BnMemory();
protected:
    virtual ~BnMemory();
};

// ----------------------------------------------------------------------------

}; // namespace android

#endif // ANDROID_IMEMORY_H
