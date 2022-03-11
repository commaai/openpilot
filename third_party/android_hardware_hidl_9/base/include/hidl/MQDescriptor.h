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

#ifndef _FMSGQ_DESCRIPTOR_H
#define _FMSGQ_DESCRIPTOR_H

#include <unistd.h>

#include <cutils/native_handle.h>
#include <hidl/HidlInternal.h>
#include <hidl/HidlSupport.h>

namespace android {
namespace hardware {

typedef uint64_t RingBufferPosition;

struct GrantorDescriptor {
    uint32_t flags __attribute__ ((aligned(4)));
    uint32_t fdIndex __attribute__ ((aligned(4)));
    uint32_t offset __attribute__ ((aligned(4)));
    uint64_t extent __attribute__ ((aligned(8)));
};

static_assert(offsetof(GrantorDescriptor, flags) == 0, "wrong offset");
static_assert(offsetof(GrantorDescriptor, fdIndex) == 4, "wrong offset");
static_assert(offsetof(GrantorDescriptor, offset) == 8, "wrong offset");
static_assert(offsetof(GrantorDescriptor, extent) == 16, "wrong offset");
static_assert(sizeof(GrantorDescriptor) == 24, "wrong size");
static_assert(__alignof(GrantorDescriptor) == 8, "wrong alignment");

enum MQFlavor : uint32_t {
  /*
   * kSynchronizedReadWrite represents the wait-free synchronized flavor of the
   * FMQ. It is intended to be have a single reader and single writer.
   * Attempts to overflow/underflow returns a failure.
   */
  kSynchronizedReadWrite = 0x01,
  /*
   * kUnsynchronizedWrite represents the flavor of FMQ where writes always
   * succeed. This flavor allows one writer and many readers. A read operation
   * can detect an overwrite and reset the read counter.
   */
  kUnsynchronizedWrite = 0x02
};

template <typename T, MQFlavor flavor>
struct MQDescriptor {
    MQDescriptor(
            const std::vector<GrantorDescriptor>& grantors,
            native_handle_t* nHandle, size_t size);

    MQDescriptor(size_t bufferSize, native_handle_t* nHandle,
                 size_t messageSize, bool configureEventFlag = false);

    MQDescriptor();
    ~MQDescriptor();

    explicit MQDescriptor(const MQDescriptor &other);
    MQDescriptor &operator=(const MQDescriptor &other) = delete;

    size_t getSize() const;

    size_t getQuantum() const;

    int32_t getFlags() const;

    bool isHandleValid() const { return mHandle != nullptr; }
    size_t countGrantors() const { return mGrantors.size(); }

    inline const ::android::hardware::hidl_vec<GrantorDescriptor> &grantors() const {
        return mGrantors;
    }

    inline ::android::hardware::hidl_vec<GrantorDescriptor> &grantors() {
        return mGrantors;
    }

    inline const ::native_handle_t *handle() const {
        return mHandle;
    }

    inline ::native_handle_t *handle() {
        return mHandle;
    }

    static const size_t kOffsetOfGrantors;
    static const size_t kOffsetOfHandle;
    enum GrantorType : int { READPTRPOS = 0, WRITEPTRPOS, DATAPTRPOS, EVFLAGWORDPOS };

    /*
     * There should at least be GrantorDescriptors for the read counter, write
     * counter and data buffer. A GrantorDescriptor for an EventFlag word is
     * not required if there is no need for blocking FMQ operations.
     */
    static constexpr int32_t kMinGrantorCount = DATAPTRPOS + 1;

    /*
     * Minimum number of GrantorDescriptors required if EventFlag support is
     * needed for blocking FMQ operations.
     */
    static constexpr int32_t kMinGrantorCountForEvFlagSupport = EVFLAGWORDPOS + 1;

    //TODO(b/34160777) Identify a better solution that supports remoting.
    static inline size_t alignToWordBoundary(size_t length) {
        constexpr size_t kAlignmentSize = 64;
        if (kAlignmentSize % __WORDSIZE != 0) {
            details::logAlwaysFatal("Incompatible word size");
        }

        /*
         * Check if alignment to word boundary would cause an overflow.
         */
        if (length > SIZE_MAX - kAlignmentSize/8 + 1) {
            details::logAlwaysFatal("Queue size too large");
        }

        return (length + kAlignmentSize/8 - 1) & ~(kAlignmentSize/8 - 1U);
    }

    static inline size_t isAlignedToWordBoundary(size_t offset) {
        constexpr size_t kAlignmentSize = 64;
        return (offset & (kAlignmentSize/8 - 1)) == 0;
    }
private:
    ::android::hardware::hidl_vec<GrantorDescriptor> mGrantors;
    ::android::hardware::details::hidl_pointer<native_handle_t> mHandle;
    uint32_t mQuantum;
    uint32_t mFlags;
};

template<typename T, MQFlavor flavor>
const size_t MQDescriptor<T, flavor>::kOffsetOfGrantors = offsetof(MQDescriptor, mGrantors);

template<typename T, MQFlavor flavor>
const size_t MQDescriptor<T, flavor>::kOffsetOfHandle = offsetof(MQDescriptor, mHandle);

/*
 * MQDescriptorSync will describe the wait-free synchronized
 * flavor of FMQ.
 */
template<typename T>
using MQDescriptorSync = MQDescriptor<T, kSynchronizedReadWrite>;

/*
 * MQDescriptorUnsync will describe the unsynchronized write
 * flavor of FMQ.
 */
template<typename T>
using MQDescriptorUnsync = MQDescriptor<T, kUnsynchronizedWrite>;

template<typename T, MQFlavor flavor>
MQDescriptor<T, flavor>::MQDescriptor(
        const std::vector<GrantorDescriptor>& grantors,
        native_handle_t* nhandle,
        size_t size)
    : mHandle(nhandle),
      mQuantum(size),
      mFlags(flavor) {
    mGrantors.resize(grantors.size());
    for (size_t i = 0; i < grantors.size(); ++i) {
        if (isAlignedToWordBoundary(grantors[i].offset) == false) {
            details::logAlwaysFatal("Grantor offsets need to be aligned");
        }
        mGrantors[i] = grantors[i];
    }
}

template<typename T, MQFlavor flavor>
MQDescriptor<T, flavor>::MQDescriptor(size_t bufferSize, native_handle_t *nHandle,
                                   size_t messageSize, bool configureEventFlag)
    : mHandle(nHandle), mQuantum(messageSize), mFlags(flavor) {
    /*
     * If configureEventFlag is true, allocate an additional spot in mGrantor
     * for containing the fd and offset for mmapping the EventFlag word.
     */
    mGrantors.resize(configureEventFlag? kMinGrantorCountForEvFlagSupport : kMinGrantorCount);

    size_t memSize[] = {
        sizeof(RingBufferPosition),  /* memory to be allocated for read pointer counter */
        sizeof(RingBufferPosition),  /* memory to be allocated for write pointer counter */
        bufferSize,                  /* memory to be allocated for data buffer */
        sizeof(std::atomic<uint32_t>)/* memory to be allocated for EventFlag word */
    };

    /*
     * Create a default grantor descriptor for read, write pointers and
     * the data buffer. fdIndex parameter is set to 0 by default and
     * the offset for each grantor is contiguous.
     */
    for (size_t grantorPos = 0, offset = 0;
         grantorPos < mGrantors.size();
         offset += memSize[grantorPos++]) {
        mGrantors[grantorPos] = {
            0 /* grantor flags */,
            0 /* fdIndex */,
            static_cast<uint32_t>(alignToWordBoundary(offset)),
            memSize[grantorPos]
        };
    }
}

template<typename T, MQFlavor flavor>
MQDescriptor<T, flavor>::MQDescriptor(const MQDescriptor<T, flavor> &other)
    : mGrantors(other.mGrantors),
      mHandle(nullptr),
      mQuantum(other.mQuantum),
      mFlags(other.mFlags) {
    if (other.mHandle != nullptr) {
        mHandle = native_handle_create(
                other.mHandle->numFds, other.mHandle->numInts);

        for (int i = 0; i < other.mHandle->numFds; ++i) {
            mHandle->data[i] = dup(other.mHandle->data[i]);
        }

        memcpy(&mHandle->data[other.mHandle->numFds],
               &other.mHandle->data[other.mHandle->numFds],
               other.mHandle->numInts * sizeof(int));
    }
}

template<typename T, MQFlavor flavor>
MQDescriptor<T, flavor>::MQDescriptor() : MQDescriptor(
        std::vector<android::hardware::GrantorDescriptor>(),
        nullptr /* nHandle */,
        0 /* size */) {}

template<typename T, MQFlavor flavor>
MQDescriptor<T, flavor>::~MQDescriptor() {
    if (mHandle != nullptr) {
        native_handle_close(mHandle);
        native_handle_delete(mHandle);
    }
}

template<typename T, MQFlavor flavor>
size_t MQDescriptor<T, flavor>::getSize() const {
  return mGrantors[DATAPTRPOS].extent;
}

template<typename T, MQFlavor flavor>
size_t MQDescriptor<T, flavor>::getQuantum() const { return mQuantum; }

template<typename T, MQFlavor flavor>
int32_t MQDescriptor<T, flavor>::getFlags() const { return mFlags; }

template<typename T, MQFlavor flavor>
std::string toString(const MQDescriptor<T, flavor> &q) {
    std::string os;
    if (flavor & kSynchronizedReadWrite) {
        os += "fmq_sync";
    }
    if (flavor & kUnsynchronizedWrite) {
        os += "fmq_unsync";
    }
    os += " {"
       + toString(q.grantors().size()) + " grantor(s), "
       + "size = " + toString(q.getSize())
       + ", .handle = " + toString(q.handle())
       + ", .quantum = " + toString(q.getQuantum()) + "}";
    return os;
}

}  // namespace hardware
}  // namespace android

#endif  // FMSGQ_DESCRIPTOR_H
