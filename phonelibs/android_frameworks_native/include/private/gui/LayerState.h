/*
 * Copyright (C) 2008 The Android Open Source Project
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

#ifndef ANDROID_SF_LAYER_STATE_H
#define ANDROID_SF_LAYER_STATE_H

#include <stdint.h>
#include <sys/types.h>

#include <utils/Errors.h>

#include <ui/Region.h>
#include <ui/Rect.h>

namespace android {

class Parcel;
class ISurfaceComposerClient;

/*
 * Used to communicate layer information between SurfaceFlinger and its clients.
 */
struct layer_state_t {


    enum {
        eLayerHidden        = 0x01,     // SURFACE_HIDDEN in SurfaceControl.java
        eLayerOpaque        = 0x02,     // SURFACE_OPAQUE
        eLayerSecure        = 0x80,     // SECURE
    };

    enum {
        ePositionChanged            = 0x00000001,
        eLayerChanged               = 0x00000002,
        eSizeChanged                = 0x00000004,
        eAlphaChanged               = 0x00000008,
        eMatrixChanged              = 0x00000010,
        eTransparentRegionChanged   = 0x00000020,
        eFlagsChanged               = 0x00000040,
        eLayerStackChanged          = 0x00000080,
        eCropChanged                = 0x00000100,
        eBlurChanged                = 0x00400000,
        eBlurMaskSurfaceChanged     = 0x00800000,
        eBlurMaskSamplingChanged    = 0x01000000,
        eBlurMaskAlphaThresholdChanged = 0x02000000,
    };

    layer_state_t()
        :   what(0),
            x(0), y(0), z(0), w(0), h(0), layerStack(0), blur(0),
            blurMaskSampling(0), blurMaskAlphaThreshold(0), alpha(0), flags(0), mask(0),
            reserved(0)
    {
        matrix.dsdx = matrix.dtdy = 1.0f;
        matrix.dsdy = matrix.dtdx = 0.0f;
        crop.makeInvalid();
    }

    status_t    write(Parcel& output) const;
    status_t    read(const Parcel& input);

            struct matrix22_t {
                float   dsdx;
                float   dtdx;
                float   dsdy;
                float   dtdy;
            };
            sp<IBinder>     surface;
            uint32_t        what;
            float           x;
            float           y;
            uint32_t        z;
            uint32_t        w;
            uint32_t        h;
            uint32_t        layerStack;
            float           blur;
            sp<IBinder>     blurMaskSurface;
            uint32_t        blurMaskSampling;
            float           blurMaskAlphaThreshold;
            float           alpha;
            uint8_t         flags;
            uint8_t         mask;
            uint8_t         reserved;
            matrix22_t      matrix;
            Rect            crop;
            // non POD must be last. see write/read
            Region          transparentRegion;
};

struct ComposerState {
    sp<ISurfaceComposerClient> client;
    layer_state_t state;
    status_t    write(Parcel& output) const;
    status_t    read(const Parcel& input);
};

struct DisplayState {

    enum {
        eOrientationDefault     = 0,
        eOrientation90          = 1,
        eOrientation180         = 2,
        eOrientation270         = 3,
        eOrientationUnchanged   = 4,
        eOrientationSwapMask    = 0x01
    };

    enum {
        eSurfaceChanged             = 0x01,
        eLayerStackChanged          = 0x02,
        eDisplayProjectionChanged   = 0x04,
        eDisplaySizeChanged         = 0x08
    };

    uint32_t what;
    sp<IBinder> token;
    sp<IGraphicBufferProducer> surface;
    uint32_t layerStack;
    uint32_t orientation;
    Rect viewport;
    Rect frame;
    uint32_t width, height;
    status_t write(Parcel& output) const;
    status_t read(const Parcel& input);
};

}; // namespace android

#endif // ANDROID_SF_LAYER_STATE_H

