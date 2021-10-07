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

#if defined(__pnacl__) || defined(__CLR_VER) || \
    (defined(__i386__) && !defined(__SSE2__))
#define LIBYUV_DISABLE_X86
#endif
// MemorySanitizer does not support assembly code yet. http://crbug.com/344505
#if defined(__has_feature)
#if __has_feature(memory_sanitizer)
#define LIBYUV_DISABLE_X86
#endif
#endif

// Visual C 2012 required for AVX2.
#if defined(_M_IX86) && !defined(__clang__) && \
    defined(_MSC_VER) && _MSC_VER >= 1700
#define VISUALC_HAS_AVX2 1
#endif  // VisualStudio >= 2012

// clang >= 3.4.0 required for AVX2.
#if defined(__clang__) && (defined(__x86_64__) || defined(__i386__))
#if (__clang_major__ > 3) || (__clang_major__ == 3 && (__clang_minor__ >= 4))
#define CLANG_HAS_AVX2 1
#endif  // clang >= 3.4
#endif  // __clang__

#if !defined(LIBYUV_DISABLE_X86) && \
    defined(_M_IX86) && (defined(VISUALC_HAS_AVX2) || defined(CLANG_HAS_AVX2))
#define HAS_HASHDJB2_AVX2
#endif

// The following are available for Visual C and GCC:
#if !defined(LIBYUV_DISABLE_X86) && \
    (defined(__x86_64__) || (defined(__i386__) || defined(_M_IX86)))
#define HAS_HASHDJB2_SSE41
#define HAS_SUMSQUAREERROR_SSE2
#endif

// The following are available for Visual C and clangcl 32 bit:
#if !defined(LIBYUV_DISABLE_X86) && defined(_M_IX86) && \
    (defined(VISUALC_HAS_AVX2) || defined(CLANG_HAS_AVX2))
#define HAS_HASHDJB2_AVX2
#define HAS_SUMSQUAREERROR_AVX2
#endif

// The following are available for Neon:
#if !defined(LIBYUV_DISABLE_NEON) && \
    (defined(__ARM_NEON__) || defined(LIBYUV_NEON) || defined(__aarch64__))
#define HAS_SUMSQUAREERROR_NEON
#endif

uint32 SumSquareError_C(const uint8* src_a, const uint8* src_b, int count);
uint32 SumSquareError_SSE2(const uint8* src_a, const uint8* src_b, int count);
uint32 SumSquareError_AVX2(const uint8* src_a, const uint8* src_b, int count);
uint32 SumSquareError_NEON(const uint8* src_a, const uint8* src_b, int count);

uint32 HashDjb2_C(const uint8* src, int count, uint32 seed);
uint32 HashDjb2_SSE41(const uint8* src, int count, uint32 seed);
uint32 HashDjb2_AVX2(const uint8* src, int count, uint32 seed);

#ifdef __cplusplus
}  // extern "C"
}  // namespace libyuv
#endif

#endif  // INCLUDE_LIBYUV_COMPARE_ROW_H_
