/*
 *  Copyright 2013 The LibYuv Project Authors. All rights reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS. All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#ifndef INCLUDE_LIBYUV_COMPARE_ROW_H_
#define INCLUDE_LIBYUV_COMPARE_ROW_H_

#include "libyuv/basic_types.h"

#ifdef __cplusplus
namespace libyuv {
extern "C" {
#endif

#if defined(__pnacl__) || defined(__CLR_VER) ||            \
    (defined(__native_client__) && defined(__x86_64__)) || \
    (defined(__i386__) && !defined(__SSE__) && !defined(__clang__))
#define LIBYUV_DISABLE_X86
#endif
#if defined(__native_client__)
#define LIBYUV_DISABLE_NEON
#endif
// MemorySanitizer does not support assembly code yet. http://crbug.com/344505
#if defined(__has_feature)
#if __has_feature(memory_sanitizer)
#define LIBYUV_DISABLE_X86
#endif
#endif
// Visual C 2012 required for AVX2.
#if defined(_M_IX86) && !defined(__clang__) && defined(_MSC_VER) && \
    _MSC_VER >= 1700
#define VISUALC_HAS_AVX2 1
#endif  // VisualStudio >= 2012

// clang >= 3.4.0 required for AVX2.
#if defined(__clang__) && (defined(__x86_64__) || defined(__i386__))
#if (__clang_major__ > 3) || (__clang_major__ == 3 && (__clang_minor__ >= 4))
#define CLANG_HAS_AVX2 1
#endif  // clang >= 3.4
#endif  // __clang__

// The following are available for Visual C and GCC:
#if !defined(LIBYUV_DISABLE_X86) && \
    (defined(__x86_64__) || defined(__i386__) || defined(_M_IX86))
#define HAS_HASHDJB2_SSE41
#define HAS_SUMSQUAREERROR_SSE2
#define HAS_HAMMINGDISTANCE_SSE42
#endif

// The following are available for Visual C and clangcl 32 bit:
#if !defined(LIBYUV_DISABLE_X86) && defined(_M_IX86) && defined(_MSC_VER) && \
    !defined(__clang__) &&                                                   \
    (defined(VISUALC_HAS_AVX2) || defined(CLANG_HAS_AVX2))
#define HAS_HASHDJB2_AVX2
#define HAS_SUMSQUAREERROR_AVX2
#endif

// The following are available for GCC and clangcl:
#if !defined(LIBYUV_DISABLE_X86) && (defined(__x86_64__) || defined(__i386__))
#define HAS_HAMMINGDISTANCE_SSSE3
#endif

// The following are available for GCC and clangcl:
#if !defined(LIBYUV_DISABLE_X86) && defined(CLANG_HAS_AVX2) && \
    (defined(__x86_64__) || defined(__i386__))
#define HAS_HAMMINGDISTANCE_AVX2
#endif

// The following are available for Neon:
#if !defined(LIBYUV_DISABLE_NEON) && \
    (defined(__ARM_NEON__) || defined(LIBYUV_NEON) || defined(__aarch64__))
#define HAS_SUMSQUAREERROR_NEON
#define HAS_HAMMINGDISTANCE_NEON
#endif

#if !defined(LIBYUV_DISABLE_MSA) && defined(__mips_msa)
#define HAS_HAMMINGDISTANCE_MSA
#define HAS_SUMSQUAREERROR_MSA
#endif

uint32_t HammingDistance_C(const uint8_t* src_a,
                           const uint8_t* src_b,
                           int count);
uint32_t HammingDistance_SSE42(const uint8_t* src_a,
                               const uint8_t* src_b,
                               int count);
uint32_t HammingDistance_SSSE3(const uint8_t* src_a,
                               const uint8_t* src_b,
                               int count);
uint32_t HammingDistance_AVX2(const uint8_t* src_a,
                              const uint8_t* src_b,
                              int count);
uint32_t HammingDistance_NEON(const uint8_t* src_a,
                              const uint8_t* src_b,
                              int count);
uint32_t HammingDistance_MSA(const uint8_t* src_a,
                             const uint8_t* src_b,
                             int count);
uint32_t SumSquareError_C(const uint8_t* src_a,
                          const uint8_t* src_b,
                          int count);
uint32_t SumSquareError_SSE2(const uint8_t* src_a,
                             const uint8_t* src_b,
                             int count);
uint32_t SumSquareError_AVX2(const uint8_t* src_a,
                             const uint8_t* src_b,
                             int count);
uint32_t SumSquareError_NEON(const uint8_t* src_a,
                             const uint8_t* src_b,
                             int count);
uint32_t SumSquareError_MSA(const uint8_t* src_a,
                            const uint8_t* src_b,
                            int count);

uint32_t HashDjb2_C(const uint8_t* src, int count, uint32_t seed);
uint32_t HashDjb2_SSE41(const uint8_t* src, int count, uint32_t seed);
uint32_t HashDjb2_AVX2(const uint8_t* src, int count, uint32_t seed);

#ifdef __cplusplus
}  // extern "C"
}  // namespace libyuv
#endif

#endif  // INCLUDE_LIBYUV_COMPARE_ROW_H_
