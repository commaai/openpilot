/*
 * Copyright (C) 2015 The Android Open Source Project
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

#ifndef ANDROID_IP_PREFIX_H
#define ANDROID_IP_PREFIX_H

#ifndef __ANDROID_VNDK__

#include <netinet/in.h>

#include <binder/Parcelable.h>
#include <utils/String16.h>
#include <utils/StrongPointer.h>

namespace android {

namespace net {

/*
 * C++ implementation of the Java class android.net.IpPrefix
 */
class IpPrefix : public Parcelable {
public:
    IpPrefix() = default;
    virtual ~IpPrefix() = default;
    IpPrefix(const IpPrefix& prefix) = default;

    IpPrefix(const struct in6_addr& addr, int32_t plen):
        mUnion(addr), mPrefixLength(plen), mIsIpv6(true) { }

    IpPrefix(const struct in_addr& addr, int32_t plen):
        mUnion(addr), mPrefixLength(plen), mIsIpv6(false) { }

    bool getAddressAsIn6Addr(struct in6_addr* addr) const;
    bool getAddressAsInAddr(struct in_addr* addr) const;

    const struct in6_addr& getAddressAsIn6Addr() const;
    const struct in_addr& getAddressAsInAddr() const;

    bool isIpv6() const;
    bool isIpv4() const;

    int32_t getPrefixLength() const;

    void setAddress(const struct in6_addr& addr);
    void setAddress(const struct in_addr& addr);

    void setPrefixLength(int32_t prefix);

    friend bool operator==(const IpPrefix& lhs, const IpPrefix& rhs);

    friend bool operator!=(const IpPrefix& lhs, const IpPrefix& rhs) {
        return !(lhs == rhs);
    }

public:
    // Overrides
    status_t writeToParcel(Parcel* parcel) const override;
    status_t readFromParcel(const Parcel* parcel) override;

private:
    union InternalUnion {
        InternalUnion() = default;
        InternalUnion(const struct in6_addr &addr):mIn6Addr(addr) { };
        InternalUnion(const struct in_addr &addr):mInAddr(addr) { };
        struct in6_addr mIn6Addr;
        struct in_addr mInAddr;
    } mUnion;
    int32_t mPrefixLength;
    bool mIsIpv6;
};

}  // namespace net

}  // namespace android

#else // __ANDROID_VNDK__
#error "This header is not visible to vendors"
#endif // __ANDROID_VNDK__

#endif  // ANDROID_IP_PREFIX_H
