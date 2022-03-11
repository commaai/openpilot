#ifndef HIDL_GENERATED_ANDROID_HARDWARE_GRAPHICS_BUFFERQUEUE_V1_0_BNHWPRODUCERLISTENER_H
#define HIDL_GENERATED_ANDROID_HARDWARE_GRAPHICS_BUFFERQUEUE_V1_0_BNHWPRODUCERLISTENER_H

#include <android/hardware/graphics/bufferqueue/1.0/IHwProducerListener.h>

namespace android {
namespace hardware {
namespace graphics {
namespace bufferqueue {
namespace V1_0 {

struct BnHwProducerListener : public ::android::hidl::base::V1_0::BnHwBase {
    explicit BnHwProducerListener(const ::android::sp<IProducerListener> &_hidl_impl);
    explicit BnHwProducerListener(const ::android::sp<IProducerListener> &_hidl_impl, const std::string& HidlInstrumentor_package, const std::string& HidlInstrumentor_interface);

    virtual ~BnHwProducerListener();

    ::android::status_t onTransact(
            uint32_t _hidl_code,
            const ::android::hardware::Parcel &_hidl_data,
            ::android::hardware::Parcel *_hidl_reply,
            uint32_t _hidl_flags = 0,
            TransactCallback _hidl_cb = nullptr) override;


    typedef IProducerListener Pure;

    typedef android::hardware::details::bnhw_tag _hidl_tag;

    ::android::sp<IProducerListener> getImpl() { return _hidl_mImpl; }
    // Methods from ::android::hardware::graphics::bufferqueue::V1_0::IProducerListener follow.
    static ::android::status_t _hidl_onBufferReleased(
            ::android::hidl::base::V1_0::BnHwBase* _hidl_this,
            const ::android::hardware::Parcel &_hidl_data,
            ::android::hardware::Parcel *_hidl_reply,
            TransactCallback _hidl_cb);


    static ::android::status_t _hidl_needsReleaseNotify(
            ::android::hidl::base::V1_0::BnHwBase* _hidl_this,
            const ::android::hardware::Parcel &_hidl_data,
            ::android::hardware::Parcel *_hidl_reply,
            TransactCallback _hidl_cb);



private:
    // Methods from ::android::hardware::graphics::bufferqueue::V1_0::IProducerListener follow.

    // Methods from ::android::hidl::base::V1_0::IBase follow.
    ::android::hardware::Return<void> ping();
    using getDebugInfo_cb = ::android::hidl::base::V1_0::IBase::getDebugInfo_cb;
    ::android::hardware::Return<void> getDebugInfo(getDebugInfo_cb _hidl_cb);

    ::android::sp<IProducerListener> _hidl_mImpl;
};

}  // namespace V1_0
}  // namespace bufferqueue
}  // namespace graphics
}  // namespace hardware
}  // namespace android

#endif  // HIDL_GENERATED_ANDROID_HARDWARE_GRAPHICS_BUFFERQUEUE_V1_0_BNHWPRODUCERLISTENER_H
