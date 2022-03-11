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

#pragma once

#include <android-base/unique_fd.h>
#include <binder/Parcelable.h>
#include <utils/Errors.h>

namespace android {

class Parcel;

namespace gui {

class BitTube : public Parcelable {
public:
    // creates an uninitialized BitTube (to unparcel into)
    BitTube() = default;

    // creates a BitTube with a a specified send and receive buffer size
    explicit BitTube(size_t bufsize);

    // creates a BitTube with a default (4KB) send buffer
    struct DefaultSizeType {};
    static constexpr DefaultSizeType DefaultSize{};
    explicit BitTube(DefaultSizeType);

    explicit BitTube(const Parcel& data);

    virtual ~BitTube() = default;

    // check state after construction
    status_t initCheck() const;

    // get receive file-descriptor
    int getFd() const;

    // get the send file-descriptor.
    int getSendFd() const;

    // moves the receive file descriptor out of this BitTube
    base::unique_fd moveReceiveFd();

    // resets this BitTube's receive file descriptor to receiveFd
    void setReceiveFd(base::unique_fd&& receiveFd);

    // send objects (sized blobs). All objects are guaranteed to be written or the call fails.
    template <typename T>
    static ssize_t sendObjects(BitTube* tube, T const* events, size_t count) {
        return sendObjects(tube, events, count, sizeof(T));
    }

    // receive objects (sized blobs). If the receiving buffer isn't large enough, excess messages
    // are silently discarded.
    template <typename T>
    static ssize_t recvObjects(BitTube* tube, T* events, size_t count) {
        return recvObjects(tube, events, count, sizeof(T));
    }

    // implement the Parcelable protocol. Only parcels the receive file descriptor
    status_t writeToParcel(Parcel* reply) const;
    status_t readFromParcel(const Parcel* parcel);

private:
    void init(size_t rcvbuf, size_t sndbuf);

    // send a message. The write is guaranteed to send the whole message or fail.
    ssize_t write(void const* vaddr, size_t size);

    // receive a message. the passed buffer must be at least as large as the write call used to send
    // the message, excess data is silently discarded.
    ssize_t read(void* vaddr, size_t size);

    base::unique_fd mSendFd;
    mutable base::unique_fd mReceiveFd;

    static ssize_t sendObjects(BitTube* tube, void const* events, size_t count, size_t objSize);

    static ssize_t recvObjects(BitTube* tube, void* events, size_t count, size_t objSize);
};

} // namespace gui
} // namespace android
