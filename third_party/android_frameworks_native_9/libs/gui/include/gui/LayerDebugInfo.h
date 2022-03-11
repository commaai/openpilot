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

#pragma once

#include <binder/Parcelable.h>

#include <ui/PixelFormat.h>
#include <ui/Region.h>

#include <string>
#include <math/vec4.h>

namespace android {

/* Class for transporting debug info from SurfaceFlinger to authorized
 * recipients.  The class is intended to be a data container. There are
 * no getters or setters.
 */
class LayerDebugInfo : public Parcelable {
public:
    LayerDebugInfo() = default;
    LayerDebugInfo(const LayerDebugInfo&) = default;
    virtual ~LayerDebugInfo() = default;

    virtual status_t writeToParcel(Parcel* parcel) const;
    virtual status_t readFromParcel(const Parcel* parcel);

    std::string mName = std::string("NOT FILLED");
    std::string mParentName = std::string("NOT FILLED");
    std::string mType = std::string("NOT FILLED");
    Region mTransparentRegion = Region::INVALID_REGION;
    Region mVisibleRegion = Region::INVALID_REGION;
    Region mSurfaceDamageRegion = Region::INVALID_REGION;
    uint32_t mLayerStack = 0;
    float mX = 0.f;
    float mY = 0.f;
    uint32_t mZ = 0 ;
    int32_t mWidth = -1;
    int32_t mHeight = -1;
    Rect mCrop = Rect::INVALID_RECT;
    Rect mFinalCrop = Rect::INVALID_RECT;
    half4 mColor = half4(1.0_hf, 1.0_hf, 1.0_hf, 0.0_hf);
    uint32_t mFlags = 0;
    PixelFormat mPixelFormat = PIXEL_FORMAT_NONE;
    android_dataspace mDataSpace = HAL_DATASPACE_UNKNOWN;
    // Row-major transform matrix (SurfaceControl::setMatrix())
    float mMatrix[2][2] = {{0.f, 0.f}, {0.f, 0.f}};
    int32_t mActiveBufferWidth = -1;
    int32_t mActiveBufferHeight = -1;
    int32_t mActiveBufferStride = 0;
    PixelFormat mActiveBufferFormat = PIXEL_FORMAT_NONE;
    int32_t mNumQueuedFrames = -1;
    bool mRefreshPending = false;
    bool mIsOpaque = false;
    bool mContentDirty = false;
};

std::string to_string(const LayerDebugInfo& info);

} // namespace android
