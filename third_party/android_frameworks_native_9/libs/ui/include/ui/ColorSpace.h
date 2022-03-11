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

#ifndef ANDROID_UI_COLOR_SPACE
#define ANDROID_UI_COLOR_SPACE

#include <array>
#include <cmath>
#include <functional>
#include <memory>
#include <string>

#include <math/mat3.h>
#include <math/scalar.h>
#include <math/vec2.h>
#include <math/vec3.h>

namespace android {

class ColorSpace {
public:
    typedef std::function<float(float)> transfer_function;
    typedef std::function<float(float)> clamping_function;

    struct TransferParameters {
        float g = 0.0f;
        float a = 0.0f;
        float b = 0.0f;
        float c = 0.0f;
        float d = 0.0f;
        float e = 0.0f;
        float f = 0.0f;
    };

    /**
     * Creates a named color space with the specified RGB->XYZ
     * conversion matrix. The white point and primaries will be
     * computed from the supplied matrix.
     *
     * The default transfer functions are a linear response x->x
     * and the default clamping function is a simple saturate
     * (clamp(x, 0, 1)).
     */
    ColorSpace(
            const std::string& name,
            const mat3& rgbToXYZ,
            transfer_function OETF = linearResponse,
            transfer_function EOTF = linearResponse,
            clamping_function clamper = saturate<float>
    ) noexcept;

    /**
     * Creates a named color space with the specified RGB->XYZ
     * conversion matrix. The white point and primaries will be
     * computed from the supplied matrix.
     *
     * The transfer functions are defined by the set of supplied
     * transfer parameters. The default clamping function is a
     * simple saturate (clamp(x, 0, 1)).
     */
    ColorSpace(
            const std::string& name,
            const mat3& rgbToXYZ,
            const TransferParameters parameters,
            clamping_function clamper = saturate<float>
    ) noexcept;

    /**
     * Creates a named color space with the specified RGB->XYZ
     * conversion matrix. The white point and primaries will be
     * computed from the supplied matrix.
     *
     * The transfer functions are defined by a simple gamma value.
     * The default clamping function is a saturate (clamp(x, 0, 1)).
     */
    ColorSpace(
            const std::string& name,
            const mat3& rgbToXYZ,
            float gamma,
            clamping_function clamper = saturate<float>
    ) noexcept;

    /**
     * Creates a named color space with the specified primaries
     * and white point. The RGB<>XYZ conversion matrices are
     * computed from the primaries and white point.
     *
     * The default transfer functions are a linear response x->x
     * and the default clamping function is a simple saturate
     * (clamp(x, 0, 1)).
     */
    ColorSpace(
            const std::string& name,
            const std::array<float2, 3>& primaries,
            const float2& whitePoint,
            transfer_function OETF = linearResponse,
            transfer_function EOTF = linearResponse,
            clamping_function clamper = saturate<float>
    ) noexcept;

    /**
     * Creates a named color space with the specified primaries
     * and white point. The RGB<>XYZ conversion matrices are
     * computed from the primaries and white point.
     *
     * The transfer functions are defined by the set of supplied
     * transfer parameters. The default clamping function is a
     * simple saturate (clamp(x, 0, 1)).
     */
    ColorSpace(
            const std::string& name,
            const std::array<float2, 3>& primaries,
            const float2& whitePoint,
            const TransferParameters parameters,
            clamping_function clamper = saturate<float>
    ) noexcept;

    /**
     * Creates a named color space with the specified primaries
     * and white point. The RGB<>XYZ conversion matrices are
     * computed from the primaries and white point.
     *
     * The transfer functions are defined by a single gamma value.
     * The default clamping function is a saturate (clamp(x, 0, 1)).
     */
    ColorSpace(
            const std::string& name,
            const std::array<float2, 3>& primaries,
            const float2& whitePoint,
            float gamma,
            clamping_function clamper = saturate<float>
    ) noexcept;

    ColorSpace() noexcept = delete;

    /**
     * Encodes the supplied RGB value using this color space's
     * opto-electronic transfer function.
     */
    constexpr float3 fromLinear(const float3& v) const noexcept {
        return apply(v, mOETF);
    }

    /**
     * Decodes the supplied RGB value using this color space's
     * electro-optical transfer function.
     */
    constexpr float3 toLinear(const float3& v) const noexcept {
        return apply(v, mEOTF);
    }

    /**
     * Converts the supplied XYZ value to RGB. The returned value
     * is encoded with this color space's opto-electronic transfer
     * function and clamped by this color space's clamping function.
     */
    constexpr float3 xyzToRGB(const float3& xyz) const noexcept {
        return apply(fromLinear(mXYZtoRGB * xyz), mClamper);
    }

    /**
     * Converts the supplied RGB value to XYZ. The input RGB value
     * is decoded using this color space's electro-optical function
     * before being converted to XYZ.
     */
    constexpr float3 rgbToXYZ(const float3& rgb) const noexcept {
        return mRGBtoXYZ * toLinear(rgb);
    }

    constexpr const std::string& getName() const noexcept {
        return mName;
    }

    constexpr const mat3& getRGBtoXYZ() const noexcept {
        return mRGBtoXYZ;
    }

    constexpr const mat3& getXYZtoRGB() const noexcept {
        return mXYZtoRGB;
    }

    constexpr const transfer_function& getOETF() const noexcept {
        return mOETF;
    }

    constexpr const transfer_function& getEOTF() const noexcept {
        return mEOTF;
    }

    constexpr const clamping_function& getClamper() const noexcept {
        return mClamper;
    }

    constexpr const std::array<float2, 3>& getPrimaries() const noexcept {
        return mPrimaries;
    }

    constexpr const float2& getWhitePoint() const noexcept {
        return mWhitePoint;
    }

    constexpr const TransferParameters& getTransferParameters() const noexcept {
        return mParameters;
    }

    /**
     * Converts the supplied XYZ value to xyY.
     */
    static constexpr float2 xyY(const float3& XYZ) {
        return XYZ.xy / dot(XYZ, float3{1});
    }

    /**
     * Converts the supplied xyY value to XYZ.
     */
    static constexpr float3 XYZ(const float3& xyY) {
        return float3{(xyY.x * xyY.z) / xyY.y, xyY.z, ((1 - xyY.x - xyY.y) * xyY.z) / xyY.y};
    }

    static const ColorSpace sRGB();
    static const ColorSpace linearSRGB();
    static const ColorSpace extendedSRGB();
    static const ColorSpace linearExtendedSRGB();
    static const ColorSpace NTSC();
    static const ColorSpace BT709();
    static const ColorSpace BT2020();
    static const ColorSpace AdobeRGB();
    static const ColorSpace ProPhotoRGB();
    static const ColorSpace DisplayP3();
    static const ColorSpace DCIP3();
    static const ColorSpace ACES();
    static const ColorSpace ACEScg();

    // Creates a NxNxN 3D LUT, where N is the specified size (min=2, max=256)
    // The 3D lookup coordinates map to the RGB components: u=R, v=G, w=B
    // The generated 3D LUT is meant to be used as a 3D texture and its Y
    // axis is thus already flipped
    // The source color space must define its values in the domain [0..1]
    // The generated LUT transforms from gamma space to gamma space
    static std::unique_ptr<float3> createLUT(uint32_t size,
            const ColorSpace& src, const ColorSpace& dst);

private:
    static constexpr mat3 computeXYZMatrix(
            const std::array<float2, 3>& primaries, const float2& whitePoint);

    static constexpr float linearResponse(float v) {
        return v;
    }

    std::string mName;

    mat3 mRGBtoXYZ;
    mat3 mXYZtoRGB;

    TransferParameters mParameters;
    transfer_function mOETF;
    transfer_function mEOTF;
    clamping_function mClamper;

    std::array<float2, 3> mPrimaries;
    float2 mWhitePoint;
};

class ColorSpaceConnector {
public:
    ColorSpaceConnector(const ColorSpace& src, const ColorSpace& dst) noexcept;

    constexpr const ColorSpace& getSource() const noexcept { return mSource; }
    constexpr const ColorSpace& getDestination() const noexcept { return mDestination; }

    constexpr const mat3& getTransform() const noexcept { return mTransform; }

    constexpr float3 transform(const float3& v) const noexcept {
        float3 linear = mSource.toLinear(apply(v, mSource.getClamper()));
        return apply(mDestination.fromLinear(mTransform * linear), mDestination.getClamper());
    }

    constexpr float3 transformLinear(const float3& v) const noexcept {
        float3 linear = apply(v, mSource.getClamper());
        return apply(mTransform * linear, mDestination.getClamper());
    }

private:
    ColorSpace mSource;
    ColorSpace mDestination;
    mat3 mTransform;
};

}; // namespace android

#endif // ANDROID_UI_COLOR_SPACE
