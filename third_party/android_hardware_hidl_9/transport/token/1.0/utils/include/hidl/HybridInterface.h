/*
 * Copyright (C) 2017 The Android Open Source Project
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

#ifndef ANDROID_HYBRIDINTERFACE_H
#define ANDROID_HYBRIDINTERFACE_H

#include <vector>
#include <mutex>

#include <binder/Parcel.h>
#include <hidl/HidlSupport.h>

/**
 * Hybrid Interfaces
 * =================
 *
 * A hybrid interface is a binder interface that
 * 1. is implemented both traditionally and as a wrapper around a hidl
 *    interface, and allows querying whether the underlying instance comes from
 *    a hidl interface or not; and
 * 2. allows efficient calls to a hidl interface (if the underlying instance
 *    comes from a hidl interface) by automatically creating the wrapper in the
 *    process that calls it.
 *
 * Terminology:
 * - `HalToken`: The type for a "token" of a hidl interface. This is defined to
 *   be compatible with `ITokenManager.hal`.
 * - `HInterface`: The base type for a hidl interface. Currently, it is defined
 *   as `::android::hidl::base::V1_0::IBase`.
 * - `HALINTERFACE`: The hidl interface that will be sent through binders.
 * - `INTERFACE`: The binder interface that will be the wrapper of
 *   `HALINTERFACE`. `INTERFACE` is supposed to be somewhat similar to
 *   `HALINTERFACE`.
 *
 * To demonstrate how this is done, here is an example. Suppose `INTERFACE` is
 * `IFoo` and `HALINTERFACE` is `HFoo`. The required steps are:
 * 1. Use DECLARE_HYBRID_META_INTERFACE instead of DECLARE_META_INTERFACE in the
 *    definition of `IFoo`. The usage is
 *        DECLARE_HYBRID_META_INTERFACE(IFoo, HFoo)
 *    inside the body of `IFoo`.
 * 2. Create a converter class that derives from
 *    `H2BConverter<HFoo, IFoo, BnFoo>`. Let us call this `H2BFoo`.
 * 3. Add the following constructor in `H2BFoo` that call the corresponding
 *    constructors in `H2BConverter`:
 *        H2BFoo(const sp<HalInterface>& base) : CBase(base) {}
 *    Note: `CBase = H2BConverter<HFoo, IFoo, BnFoo>` and `HalInterface = HFoo`
 *    are member typedefs of `H2BConverter<HFoo, IFoo, BnFoo>`, so the above
 *    line can be copied into `H2BFoo`.
 * 4. Implement `IFoo` in `H2BFoo` on top of `HFoo`. `H2BConverter` provides a
 *    protected `mBase` of type `sp<HFoo>` that can be used to access the `HFoo`
 *    instance. (There is also a public function named `getHalInterface()` that
 *    returns `mBase`.)
 * 5. Create a hardware proxy class that derives from
 *    `HpInterface<BpFoo, H2BFoo>`. Name this class `HpFoo`. (This name cannot
 *    deviate. See step 8 below.)
 * 6. Add the following constructor to `HpFoo`:
 *        HpFoo(const sp<IBinder>& base): PBase(base) {}
 *    Note: `PBase` a member typedef of `HpInterface<BpFoo, H2BFoo>` that is
 *    equal to `HpInterface<BpFoo, H2BFoo>` itself, so the above line can be
 *    copied verbatim into `HpFoo`.
 * 7. Delegate all functions in `HpFoo` that come from `IFoo` except
 *    `getHalInterface` to the protected member `mBase`,
 *    which is defined in `HpInterface<BpFoo, H2BFoo>` (hence in `HpFoo`) with
 *    type `IFoo`. (There is also a public function named `getBaseInterface()`
 *    that returns `mBase`.)
 * 8. Replace the existing `IMPLEMENT_META_INTERFACE` for INTERFACE by
 *    `IMPLEMENT_HYBRID_META_INTERFACE`. Note that this macro relies on the
 *    exact naming of `HpFoo`, where `Foo` comes from the interface name `IFoo`.
 *    An example usage is
 *        IMPLEMENT_HYBRID_META_INTERFACE(IFoo, HFoo, "example.interface.foo");
 *
 * `GETTOKEN` Template Argument
 * ============================
 *
 * Following the instructions above, `H2BConverter` and `HpInterface` would use
 * `transact()` to send over tokens, with `code` (the first argument of
 * `transact()`) equal to `DEFAULT_GET_HAL_TOKEN_TRANSACTION_CODE`. If this
 * value clashes with other values already in use in the `Bp` class, it can be
 * changed by supplying the last optional template argument to `H2BConverter`
 * and `HpInterface`.
 *
 */

namespace android {

typedef ::android::hardware::hidl_vec<uint8_t> HalToken;
typedef ::android::hidl::base::V1_0::IBase HInterface;

constexpr uint32_t DEFAULT_GET_HAL_TOKEN_TRANSACTION_CODE =
        B_PACK_CHARS('_', 'G', 'H', 'T');

sp<HInterface> retrieveHalInterface(const HalToken& token);
bool createHalToken(const sp<HInterface>& interface, HalToken* token);
bool deleteHalToken(const HalToken& token);

template <
        typename HINTERFACE,
        typename INTERFACE,
        typename BNINTERFACE,
        uint32_t GETTOKEN = DEFAULT_GET_HAL_TOKEN_TRANSACTION_CODE>
class H2BConverter : public BNINTERFACE {
public:
    typedef H2BConverter<HINTERFACE, INTERFACE, BNINTERFACE, GETTOKEN> CBase; // Converter Base
    typedef INTERFACE BaseInterface;
    typedef HINTERFACE HalInterface;
    static constexpr uint32_t GET_HAL_TOKEN = GETTOKEN;

    H2BConverter(const sp<HalInterface>& base) : mBase(base) {}
    virtual status_t onTransact(uint32_t code,
            const Parcel& data, Parcel* reply, uint32_t flags = 0);
    virtual sp<HalInterface> getHalInterface() { return mBase; }
    HalInterface* getBaseInterface() { return mBase.get(); }
    virtual status_t linkToDeath(
            const sp<IBinder::DeathRecipient>& recipient,
            void* cookie = nullptr,
            uint32_t flags = 0);
    virtual status_t unlinkToDeath(
            const wp<IBinder::DeathRecipient>& recipient,
            void* cookie = nullptr,
            uint32_t flags = 0,
            wp<IBinder::DeathRecipient>* outRecipient = nullptr);

protected:
    sp<HalInterface> mBase;
    struct Obituary : public hardware::hidl_death_recipient {
        wp<IBinder::DeathRecipient> recipient;
        void* cookie;
        uint32_t flags;
        wp<IBinder> who;
        Obituary(
                const wp<IBinder::DeathRecipient>& r,
                void* c, uint32_t f,
                const wp<IBinder>& w) :
            recipient(r), cookie(c), flags(f), who(w) {
        }
        Obituary(const Obituary& o) :
            recipient(o.recipient),
            cookie(o.cookie),
            flags(o.flags),
            who(o.who) {
        }
        Obituary& operator=(const Obituary& o) {
            recipient = o.recipient;
            cookie = o.cookie;
            flags = o.flags;
            who = o.who;
            return *this;
        }
        void serviceDied(uint64_t, const wp<HInterface>&) override {
            sp<IBinder::DeathRecipient> dr = recipient.promote();
            if (dr != nullptr) {
                dr->binderDied(who);
            }
        }
    };
    std::mutex mObituariesLock;
    std::vector<sp<Obituary> > mObituaries;
};

template <
        typename BPINTERFACE,
        typename CONVERTER,
        uint32_t GETTOKEN = DEFAULT_GET_HAL_TOKEN_TRANSACTION_CODE>
class HpInterface : public CONVERTER::BaseInterface {
public:
    typedef HpInterface<BPINTERFACE, CONVERTER, GETTOKEN> PBase; // Proxy Base
    typedef typename CONVERTER::BaseInterface BaseInterface;
    typedef typename CONVERTER::HalInterface HalInterface;
    static constexpr uint32_t GET_HAL_TOKEN = GETTOKEN;

    explicit HpInterface(const sp<IBinder>& impl);
    virtual sp<HalInterface> getHalInterface() { return mHal; }
    BaseInterface* getBaseInterface() { return mBase.get(); }

protected:
    IBinder* mImpl;
    sp<BPINTERFACE> mBp;
    sp<BaseInterface> mBase;
    sp<HalInterface> mHal;
    IBinder* onAsBinder() override { return mImpl; }
};

// ----------------------------------------------------------------------

#define DECLARE_HYBRID_META_INTERFACE(INTERFACE, HAL)                   \
    static const ::android::String16 descriptor;                        \
    static ::android::sp<I##INTERFACE> asInterface(                     \
            const ::android::sp<::android::IBinder>& obj);              \
    virtual const ::android::String16& getInterfaceDescriptor() const;  \
    I##INTERFACE();                                                     \
    virtual ~I##INTERFACE();                                            \
    virtual sp<HAL> getHalInterface();                                  \


#define IMPLEMENT_HYBRID_META_INTERFACE(INTERFACE, HAL, NAME)           \
    const ::android::String16 I##INTERFACE::descriptor(NAME);           \
    const ::android::String16&                                          \
            I##INTERFACE::getInterfaceDescriptor() const {              \
        return I##INTERFACE::descriptor;                                \
    }                                                                   \
    ::android::sp<I##INTERFACE> I##INTERFACE::asInterface(              \
            const ::android::sp<::android::IBinder>& obj)               \
    {                                                                   \
        ::android::sp<I##INTERFACE> intr;                               \
        if (obj != nullptr) {                                           \
            intr = static_cast<I##INTERFACE*>(                          \
                obj->queryLocalInterface(                               \
                        I##INTERFACE::descriptor).get());               \
            if (intr == nullptr) {                                      \
                intr = new Hp##INTERFACE(obj);                          \
            }                                                           \
        }                                                               \
        return intr;                                                    \
    }                                                                   \
    I##INTERFACE::I##INTERFACE() { }                                    \
    I##INTERFACE::~I##INTERFACE() { }                                   \
    sp<HAL> I##INTERFACE::getHalInterface() { return nullptr; }         \

// ----------------------------------------------------------------------

template <
        typename HINTERFACE,
        typename INTERFACE,
        typename BNINTERFACE,
        uint32_t GETTOKEN>
status_t H2BConverter<HINTERFACE, INTERFACE, BNINTERFACE, GETTOKEN>::
        onTransact(
        uint32_t code, const Parcel& data, Parcel* reply, uint32_t flags) {
    if (code == GET_HAL_TOKEN) {
        HalToken token;
        bool result;
        result = createHalToken(mBase, &token);
        if (!result) {
            ALOGE("H2BConverter: Failed to create HAL token.");
        }
        reply->writeBool(result);
        reply->writeByteArray(token.size(), token.data());
        return NO_ERROR;
    }
    return BNINTERFACE::onTransact(code, data, reply, flags);
}

template <
        typename HINTERFACE,
        typename INTERFACE,
        typename BNINTERFACE,
        uint32_t GETTOKEN>
status_t H2BConverter<HINTERFACE, INTERFACE, BNINTERFACE, GETTOKEN>::
        linkToDeath(
        const sp<IBinder::DeathRecipient>& recipient,
        void* cookie, uint32_t flags) {
    LOG_ALWAYS_FATAL_IF(recipient == nullptr,
            "linkToDeath(): recipient must be non-nullptr");
    {
        std::lock_guard<std::mutex> lock(mObituariesLock);
        mObituaries.push_back(new Obituary(recipient, cookie, flags, this));
        if (!mBase->linkToDeath(mObituaries.back(), 0)) {
           return DEAD_OBJECT;
        }
    }
    return NO_ERROR;
}

template <
        typename HINTERFACE,
        typename INTERFACE,
        typename BNINTERFACE,
        uint32_t GETTOKEN>
status_t H2BConverter<HINTERFACE, INTERFACE, BNINTERFACE, GETTOKEN>::
        unlinkToDeath(
        const wp<IBinder::DeathRecipient>& recipient,
        void* cookie, uint32_t flags,
        wp<IBinder::DeathRecipient>* outRecipient) {
    std::lock_guard<std::mutex> lock(mObituariesLock);
    for (auto i = mObituaries.begin(); i != mObituaries.end(); ++i) {
        if ((flags = (*i)->flags) && (
                (recipient == (*i)->recipient) ||
                ((recipient == nullptr) && (cookie == (*i)->cookie)))) {
            if (outRecipient != nullptr) {
                *outRecipient = (*i)->recipient;
            }
            bool success = mBase->unlinkToDeath(*i);
            mObituaries.erase(i);
            return success ? NO_ERROR : DEAD_OBJECT;
        }
    }
    return NAME_NOT_FOUND;
}

template <typename BPINTERFACE, typename CONVERTER, uint32_t GETTOKEN>
HpInterface<BPINTERFACE, CONVERTER, GETTOKEN>::HpInterface(
        const sp<IBinder>& impl) :
    mImpl(impl.get()),
    mBp(new BPINTERFACE(impl)) {
    mBase = mBp;
    if (mImpl->remoteBinder() == nullptr) {
        return;
    }
    Parcel data, reply;
    data.writeInterfaceToken(BaseInterface::getInterfaceDescriptor());
    if (mImpl->transact(GET_HAL_TOKEN, data, &reply) == NO_ERROR) {
        bool tokenCreated = reply.readBool();

        std::vector<uint8_t> tokenVector;
        reply.readByteVector(&tokenVector);
        HalToken token = HalToken(tokenVector);

        if (tokenCreated) {
            sp<HInterface> hBase = retrieveHalInterface(token);
            if (hBase != nullptr) {
                mHal = HalInterface::castFrom(hBase);
                if (mHal != nullptr) {
                    mBase = new CONVERTER(mHal);
                } else {
                    ALOGE("HpInterface: Wrong interface type.");
                }
            } else {
                ALOGE("HpInterface: Invalid HAL token.");
            }
            deleteHalToken(token);
        }
    }
}

// ----------------------------------------------------------------------

}; // namespace android

#endif // ANDROID_HYBRIDINTERFACE_H
