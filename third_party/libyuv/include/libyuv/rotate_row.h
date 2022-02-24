/*
 *  Copyright 2013 The LibYuv Project Authors. All rights reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS. All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#ifndef INCLUDE_LIBYUV_ROTATE_ROW_H_
#define INCLUDE_LIBYUV_ROTATE_ROW_H_

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
// The following are available for Visual C and clangcl 32 bit:
#if !defined(LIBYUV_DISABLE_X86) && defined(_M_IX86)
#define HAS_TRANSPOSEWX8_SSSE3
#define HAS_TRANSPOSEUVWX8_SSE2
#endif

// The following are available for GCC 32 or 64 bit but not NaCL for 64 bit:
#if !defined(LIBYUV_DISABLE_X86) && \
    (defined(__i386__) || (defined(__x86_64__) && !defined(__native_client__)))
#define HAS_TRANSPOSEWX8_SSSE3
#endif

// The following are available for 64 bit GCC but not NaCL:
#if !defined(LIBYUV_DISABLE_X86) && !defined(__native_client__) && \
    defined(__x86_64__)
#define HAS_TRANSPOSEWX8_FAST_SSSE3
#define HAS_TRANSPOSEUVWX8_SSE2
#endif

#if !defined(LIBYUV_DISABLE_NEON) && !defined(__native_client__) && \
    (defined(__ARM_NEON__) || defined(LIBYUV_NEON) || defined(__aarch64__))
#define HAS_TRANSPOSEWX8_NEON
#define HAS_TRANSPOSEUVWX8_NEON
#endif

#if !defined(LIBYUV_DISABLE_MIPS) && !defined(__native_client__) && \
    defined(__mips__) && \
    defined(__mips_dsp) && (__mips_dsp_rev >= 2)
#define HAS_TRANSPOSEWX8_DSPR2
#define HAS_TRANSPOSEUVWX8_DSPR2
#endif  // defined(__mips__)

void TransposeWxH_C(const uint8* src, int src_stride,
                    uint8* dst, int dst_stride, int width, int height);

void TransposeWx8_C(const uint8* src, int src_stride,
                    uint8* dst, int dst_stride, int width);
void TransposeWx8_NEON(const uint8* src, int src_stride,
                       uint8* dst, int dst_stride, int width);
void TransposeWx8_SSSE3(const uint8* src, int src_stride,
                        uint8* dst, int dst_stride, int width);
void TransposeWx8_Fast_SSSE3(const uint8* src, int src_stride,
                             uint8* dst, int dst_stride, int width);
void TransposeWx8_DSPR2(const uint8* src, int src_stride,
                        uint8* dst, int dst_stride, int width);
void TransposeWx8_Fast_DSPR2(const uint8* src, int src_stride,
                             uint8* dst, int dst_stride, int width);

void TransposeWx8_Any_NEON(const uint8* src, int src_stride,
                           uint8* dst, int dst_stride, int width);
void TransposeWx8_Any_SSSE3(const uint8* src, int src_stride,
                            uint8* dst, int dst_stride, int width);
void TransposeWx8_Fast_Any_SSSE3(const uint8* src, int src_stride,
                                 uint8* dst, int dst_stride, int width);
void TransposeWx8_Any_DSPR2(const uint8* src, int src_stride,
                            uint8* dst, int dst_stride, int width);

void TransposeUVWxH_C(const uint8* src, int src_stride,
                      uint8* dst_a, int dst_stride_a,
                      uint8* dst_b, int dst_stride_b,
                      int width, int height);

void TransposeUVWx8_C(const uint8* src, int src_stride,
                      uint8* dst_a, int dst_stride_a,
                      uint8* dst_b, int dst_stride_b, int width);
void TransposeUVWx8_SSE2(const uint8* src, int src_stride,
                         uint8* dst_a, int dst_stride_a,
                         uint8* dst_b, int dst_stride_b, int width);
void TransposeUVWx8_NEON(const uint8* src, int src_stride,
                         uint8* dst_a, int dst_stride_a,
                         uint8* dst_b, int dst_stride_b, int width);
void TransposeUVWx8_DSPR2(const uint8* src, int src_stride,
                          uint8* dst_a, int dst_stride_a,
                          uint8* dst_b, int dst_stride_b, int width);

void TransposeUVWx8_Any_SSE2(const uint8* src, int src_stride,
                             uint8* dst_a, int dst_stride_a,
                             uint8* dst_b, int dst_stride_b, int width);
void TransposeUVWx8_Any_NEON(const uint8* src, int src_stride,
                             uint8* dst_a, int dst_stride_a,
                             uint8* dst_b, int dst_stride_b, int width);
void TransposeUVWx8_Any_DSPR2(const uint8* src, int src_stride,
                              uint8* dst_a, int dst_stride_a,
                              uint8* dst_b, int dst_stride_b, int width);

#ifdef __cplusplus
}  // extern "C"
}  // namespace libyuv
#endif

#endif  // INCLUDE_LIBYUV_ROTATE_ROW_H_
