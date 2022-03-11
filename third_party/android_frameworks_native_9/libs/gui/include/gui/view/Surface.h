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

#ifndef ANDROID_GUI_VIEW_SURFACE_H
#define ANDROID_GUI_VIEW_SURFACE_H

#include <utils/Errors.h>
#include <utils/StrongPointer.h>
#include <utils/String16.h>

#include <binder/Parcelable.h>

namespace android {

class IGraphicBufferProducer;

namespace view {

/**
 * A simple holder for an IGraphicBufferProducer, to match the managed-side
 * android.view.Surface parcelable behavior.
 *
 * This implements android/view/Surface.aidl
 *
 * TODO: Convert IGraphicBufferProducer into AIDL so that it can be directly
 * used in managed Binder calls.
 */
class Surface : public Parcelable {
  public:

    String16 name;
    sp<IGraphicBufferProducer> graphicBufferProducer;

    virtual status_t writeToParcel(Parcel* parcel) const override;
    virtual status_t readFromParcel(const Parcel* parcel) override;

    // nameAlreadyWritten set to true by Surface.java, because it splits
    // Parceling itself between managed and native code, so it only wants a part
    // of the full parceling to happen on its native side.
    status_t writeToParcel(Parcel* parcel, bool nameAlreadyWritten) const;

    // nameAlreadyRead set to true by Surface.java, because it splits
    // Parceling itself between managed and native code, so it only wants a part
    // of the full parceling to happen on its native side.
    status_t readFromParcel(const Parcel* parcel, bool nameAlreadyRead);

  private:

    static String16 readMaybeEmptyString16(const Parcel* parcel);
};

} // namespace view
} // namespace android

#endif  // ANDROID_GUI_VIEW_SURFACE_H
