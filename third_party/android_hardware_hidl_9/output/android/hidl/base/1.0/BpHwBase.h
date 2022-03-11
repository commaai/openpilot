#ifndef HIDL_GENERATED_ANDROID_HIDL_BASE_V1_0_BPHWBASE_H
#define HIDL_GENERATED_ANDROID_HIDL_BASE_V1_0_BPHWBASE_H

#include <hidl/HidlTransportSupport.h>

#include <android/hidl/base/1.0/IHwBase.h>

namespace android {
namespace hidl {
namespace base {
namespace V1_0 {

struct BpHwBase : public ::android::hardware::BpInterface<IBase>, public ::android::hardware::details::HidlInstrumentor {
    explicit BpHwBase(const ::android::sp<::android::hardware::IBinder> &_hidl_impl);

    typedef IBase Pure;

    typedef android::hardware::details::bphw_tag _hidl_tag;

    virtual bool isRemote() const override { return true; }

    // Methods from ::android::hidl::base::V1_0::IBase follow.
    static ::android::hardware::Return<void>  _hidl_interfaceChain(::android::hardware::IInterface* _hidl_this, ::android::hardware::details::HidlInstrumentor *_hidl_this_instrumentor, interfaceChain_cb _hidl_cb);
    static ::android::hardware::Return<void>  _hidl_debug(::android::hardware::IInterface* _hidl_this, ::android::hardware::details::HidlInstrumentor *_hidl_this_instrumentor, const ::android::hardware::hidl_handle& fd, const ::android::hardware::hidl_vec<::android::hardware::hidl_string>& options);
    static ::android::hardware::Return<void>  _hidl_interfaceDescriptor(::android::hardware::IInterface* _hidl_this, ::android::hardware::details::HidlInstrumentor *_hidl_this_instrumentor, interfaceDescriptor_cb _hidl_cb);
    static ::android::hardware::Return<void>  _hidl_getHashChain(::android::hardware::IInterface* _hidl_this, ::android::hardware::details::HidlInstrumentor *_hidl_this_instrumentor, getHashChain_cb _hidl_cb);
    static ::android::hardware::Return<void>  _hidl_setHALInstrumentation(::android::hardware::IInterface* _hidl_this, ::android::hardware::details::HidlInstrumentor *_hidl_this_instrumentor);
    static ::android::hardware::Return<void>  _hidl_ping(::android::hardware::IInterface* _hidl_this, ::android::hardware::details::HidlInstrumentor *_hidl_this_instrumentor);
    static ::android::hardware::Return<void>  _hidl_getDebugInfo(::android::hardware::IInterface* _hidl_this, ::android::hardware::details::HidlInstrumentor *_hidl_this_instrumentor, getDebugInfo_cb _hidl_cb);
    static ::android::hardware::Return<void>  _hidl_notifySyspropsChanged(::android::hardware::IInterface* _hidl_this, ::android::hardware::details::HidlInstrumentor *_hidl_this_instrumentor);

    // Methods from ::android::hidl::base::V1_0::IBase follow.
    ::android::hardware::Return<void> interfaceChain(interfaceChain_cb _hidl_cb) override;
    ::android::hardware::Return<void> debug(const ::android::hardware::hidl_handle& fd, const ::android::hardware::hidl_vec<::android::hardware::hidl_string>& options) override;
    ::android::hardware::Return<void> interfaceDescriptor(interfaceDescriptor_cb _hidl_cb) override;
    ::android::hardware::Return<void> getHashChain(getHashChain_cb _hidl_cb) override;
    ::android::hardware::Return<void> setHALInstrumentation() override;
    ::android::hardware::Return<bool> linkToDeath(const ::android::sp<::android::hardware::hidl_death_recipient>& recipient, uint64_t cookie) override;
    ::android::hardware::Return<void> ping() override;
    ::android::hardware::Return<void> getDebugInfo(getDebugInfo_cb _hidl_cb) override;
    ::android::hardware::Return<void> notifySyspropsChanged() override;
    ::android::hardware::Return<bool> unlinkToDeath(const ::android::sp<::android::hardware::hidl_death_recipient>& recipient) override;

private:
    std::mutex _hidl_mMutex;
    std::vector<::android::sp<::android::hardware::hidl_binder_death_recipient>> _hidl_mDeathRecipients;
};

}  // namespace V1_0
}  // namespace base
}  // namespace hidl
}  // namespace android

#endif  // HIDL_GENERATED_ANDROID_HIDL_BASE_V1_0_BPHWBASE_H
