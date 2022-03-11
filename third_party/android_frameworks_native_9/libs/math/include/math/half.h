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

#pragma once

#include <stdint.h>
#include <iosfwd>
#include <limits>
#include <type_traits>

#ifndef LIKELY
#define LIKELY_DEFINED_LOCAL
#ifdef __cplusplus
#   define LIKELY( exp )    (__builtin_expect( !!(exp), true ))
#   define UNLIKELY( exp )  (__builtin_expect( !!(exp), false ))
#else
#   define LIKELY( exp )    (__builtin_expect( !!(exp), 1 ))
#   define UNLIKELY( exp )  (__builtin_expect( !!(exp), 0 ))
#endif
#endif

#if __cplusplus >= 201402L
#define CONSTEXPR constexpr
#else
#define CONSTEXPR
#endif

namespace android {

/*
 * half-float
 *
 *  1   5       10
 * +-+------+------------+
 * |s|eee.ee|mm.mmmm.mmmm|
 * +-+------+------------+
 *
 * minimum (denormal) value: 2^-24 = 5.96e-8
 * minimum (normal) value:   2^-14 = 6.10e-5
 * maximum value:            2-2^-10 = 65504
 *
 * Integers between 0 and 2048 can be represented exactly
 */
class half {
    struct fp16 {
        uint16_t bits;
        explicit constexpr fp16() noexcept : bits(0) { }
        explicit constexpr fp16(uint16_t b) noexcept : bits(b) { }
        void setS(unsigned int s) noexcept { bits = uint16_t((bits & 0x7FFF) | (s<<15)); }
        void setE(unsigned int s) noexcept { bits = uint16_t((bits & 0xE3FF) | (s<<10)); }
        void setM(unsigned int s) noexcept { bits = uint16_t((bits & 0xFC00) | (s<< 0)); }
        constexpr unsigned int getS() const noexcept { return  bits >> 15u; }
        constexpr unsigned int getE() const noexcept { return (bits >> 10u) & 0x1Fu; }
        constexpr unsigned int getM() const noexcept { return  bits         & 0x3FFu; }
    };
    struct fp32 {
        union {
            uint32_t bits;
            float fp;
        };
        explicit constexpr fp32() noexcept : bits(0) { }
        explicit constexpr fp32(float f) noexcept : fp(f) { }
        void setS(unsigned int s) noexcept { bits = uint32_t((bits & 0x7FFFFFFF) | (s<<31)); }
        void setE(unsigned int s) noexcept { bits = uint32_t((bits & 0x807FFFFF) | (s<<23)); }
        void setM(unsigned int s) noexcept { bits = uint32_t((bits & 0xFF800000) | (s<< 0)); }
        constexpr unsigned int getS() const noexcept { return  bits >> 31u; }
        constexpr unsigned int getE() const noexcept { return (bits >> 23u) & 0xFFu; }
        constexpr unsigned int getM() const noexcept { return  bits         & 0x7FFFFFu; }
    };

public:
    CONSTEXPR half(float v) noexcept : mBits(ftoh(v)) { }
    CONSTEXPR operator float() const noexcept { return htof(mBits); }

    uint16_t getBits() const noexcept { return mBits.bits; }
    unsigned int getExponent() const noexcept { return mBits.getE(); }
    unsigned int getMantissa() const noexcept { return mBits.getM(); }

private:
    friend class std::numeric_limits<half>;
    friend CONSTEXPR half operator"" _hf(long double v);

    enum Binary { binary };
    explicit constexpr half(Binary, uint16_t bits) noexcept : mBits(bits) { }
    static CONSTEXPR fp16 ftoh(float v) noexcept;
    static CONSTEXPR float htof(fp16 v) noexcept;
    fp16 mBits;
};

inline CONSTEXPR half::fp16 half::ftoh(float v) noexcept {
    fp16 out;
    fp32 in(v);
    if (UNLIKELY(in.getE() == 0xFF)) { // inf or nan
        out.setE(0x1F);
        out.setM(in.getM() ? 0x200 : 0);
    } else {
        int e = static_cast<int>(in.getE()) - 127 + 15;
        if (e >= 0x1F) {
            // overflow
            out.setE(0x31); // +/- inf
        } else if (e <= 0) {
            // underflow
            // flush to +/- 0
        } else {
            unsigned int m = in.getM();
            out.setE(uint16_t(e));
            out.setM(m >> 13);
            if (m & 0x1000) {
                // rounding
                out.bits++;
            }
        }
    }
    out.setS(in.getS());
    return out;
}

inline CONSTEXPR float half::htof(half::fp16 in) noexcept {
    fp32 out;
    if (UNLIKELY(in.getE() == 0x1F)) { // inf or nan
        out.setE(0xFF);
        out.setM(in.getM() ? 0x400000 : 0);
    } else {
        if (in.getE() == 0) {
            if (in.getM()) {
                // TODO: denormal half float, treat as zero for now
                // (it's stupid because they can be represented as regular float)
            }
        } else {
            int e = static_cast<int>(in.getE()) - 15 + 127;
            unsigned int m = in.getM();
            out.setE(uint32_t(e));
            out.setM(m << 13);
        }
    }
    out.setS(in.getS());
    return out.fp;
}

inline CONSTEXPR android::half operator"" _hf(long double v) {
    return android::half(android::half::binary, android::half::ftoh(static_cast<float>(v)).bits);
}

} // namespace android

namespace std {

template<> struct is_floating_point<android::half> : public std::true_type {};

template<>
class numeric_limits<android::half> {
public:
    typedef android::half type;

    static constexpr const bool is_specialized = true;
    static constexpr const bool is_signed = true;
    static constexpr const bool is_integer = false;
    static constexpr const bool is_exact = false;
    static constexpr const bool has_infinity = true;
    static constexpr const bool has_quiet_NaN = true;
    static constexpr const bool has_signaling_NaN = false;
    static constexpr const float_denorm_style has_denorm = denorm_absent;
    static constexpr const bool has_denorm_loss = true;
    static constexpr const bool is_iec559 = false;
    static constexpr const bool is_bounded = true;
    static constexpr const bool is_modulo = false;
    static constexpr const bool traps = false;
    static constexpr const bool tinyness_before = false;
    static constexpr const float_round_style round_style = round_indeterminate;

    static constexpr const int digits = 11;
    static constexpr const int digits10 = 3;
    static constexpr const int max_digits10 = 5;
    static constexpr const int radix = 2;
    static constexpr const int min_exponent = -13;
    static constexpr const int min_exponent10 = -4;
    static constexpr const int max_exponent = 16;
    static constexpr const int max_exponent10 = 4;

    inline static constexpr type round_error() noexcept { return android::half(android::half::binary, 0x3800); }
    inline static constexpr type min() noexcept { return android::half(android::half::binary, 0x0400); }
    inline static constexpr type max() noexcept { return android::half(android::half::binary, 0x7bff); }
    inline static constexpr type lowest() noexcept { return android::half(android::half::binary, 0xfbff); }
    inline static constexpr type epsilon() noexcept { return android::half(android::half::binary, 0x1400); }
    inline static constexpr type infinity() noexcept { return android::half(android::half::binary, 0x7c00); }
    inline static constexpr type quiet_NaN() noexcept { return android::half(android::half::binary, 0x7fff); }
    inline static constexpr type denorm_min() noexcept { return android::half(android::half::binary, 0x0001); }
    inline static constexpr type signaling_NaN() noexcept { return android::half(android::half::binary, 0x7dff); }
};

} // namespace std

#ifdef LIKELY_DEFINED_LOCAL
#undef LIKELY_DEFINED_LOCAL
#undef LIKELY
#undef UNLIKELY
#endif // LIKELY_DEFINED_LOCAL

#undef CONSTEXPR
