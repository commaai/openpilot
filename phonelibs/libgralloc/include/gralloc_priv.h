/*
 * Copyright (C) 2008 The Android Open Source Project
 * Copyright (c) 2011-2015, The Linux Foundation. All rights reserved.
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

#ifndef GRALLOC_PRIV_H_
#define GRALLOC_PRIV_H_

#include <stdint.h>
#include <limits.h>
#include <sys/cdefs.h>
#include <hardware/gralloc.h>
#include <pthread.h>
#include <errno.h>
#include <unistd.h>

#include <cutils/native_handle.h>

#include <cutils/log.h>

#define ROUND_UP_PAGESIZE(x) ( (((unsigned long)(x)) + PAGE_SIZE-1)  & \
                               (~(PAGE_SIZE-1)) )

/* Gralloc usage bits indicating the type of allocation that should be used */
/* SYSTEM heap comes from kernel vmalloc (ION_SYSTEM_HEAP_ID)
 * is cached by default and
 * is not secured */

/* GRALLOC_USAGE_PRIVATE_0 is unused */

/* Non linear, Universal Bandwidth Compression */
#define GRALLOC_USAGE_PRIVATE_ALLOC_UBWC      GRALLOC_USAGE_PRIVATE_1

/* IOMMU heap comes from manually allocated pages, can be cached/uncached,
 * is not secured */
#define GRALLOC_USAGE_PRIVATE_IOMMU_HEAP      GRALLOC_USAGE_PRIVATE_2

/* MM heap is a carveout heap for video, can be secured */
#define GRALLOC_USAGE_PRIVATE_MM_HEAP         GRALLOC_USAGE_PRIVATE_3

/* ADSP heap is a carveout heap, is not secured */
#define GRALLOC_USAGE_PRIVATE_ADSP_HEAP       0x01000000

/* Set this for allocating uncached memory (using O_DSYNC),
 * cannot be used with noncontiguous heaps */
#define GRALLOC_USAGE_PRIVATE_UNCACHED        0x02000000

/* Buffer content should be displayed on an primary display only */
#define GRALLOC_USAGE_PRIVATE_INTERNAL_ONLY   0x04000000

/* Buffer content should be displayed on an external display only */
#define GRALLOC_USAGE_PRIVATE_EXTERNAL_ONLY   0x08000000

/* This flag is set for WFD usecase */
#define GRALLOC_USAGE_PRIVATE_WFD             0x00200000

/* CAMERA heap is a carveout heap for camera, is not secured */
#define GRALLOC_USAGE_PRIVATE_CAMERA_HEAP     0x00400000

/* This flag is used for SECURE display usecase */
#define GRALLOC_USAGE_PRIVATE_SECURE_DISPLAY  0x00800000

/* define Gralloc perform */
#define GRALLOC_MODULE_PERFORM_CREATE_HANDLE_FROM_BUFFER 1
// This will be used by the graphics drivers to know if certain features
// are defined in this display HAL.
// Ex: Newer GFX libraries + Older Display HAL
#define GRALLOC_MODULE_PERFORM_GET_STRIDE 2
#define GRALLOC_MODULE_PERFORM_GET_CUSTOM_STRIDE_FROM_HANDLE 3
#define GRALLOC_MODULE_PERFORM_GET_CUSTOM_STRIDE_AND_HEIGHT_FROM_HANDLE 4
#define GRALLOC_MODULE_PERFORM_GET_ATTRIBUTES 5
#define GRALLOC_MODULE_PERFORM_GET_COLOR_SPACE_FROM_HANDLE 6
#define GRALLOC_MODULE_PERFORM_GET_YUV_PLANE_INFO 7
#define GRALLOC_MODULE_PERFORM_GET_MAP_SECURE_BUFFER_INFO 8
#define GRALLOC_MODULE_PERFORM_GET_UBWC_FLAG 9
#define GRALLOC_MODULE_PERFORM_GET_RGB_DATA_ADDRESS 10
#define GRALLOC_MODULE_PERFORM_GET_IGC 11
#define GRALLOC_MODULE_PERFORM_SET_IGC 12
#define GRALLOC_MODULE_PERFORM_SET_SINGLE_BUFFER_MODE 13

/* OEM specific HAL formats */

#define HAL_PIXEL_FORMAT_RGBA_5551               6
#define HAL_PIXEL_FORMAT_RGBA_4444               7
#define HAL_PIXEL_FORMAT_NV12_ENCODEABLE         0x102
#define HAL_PIXEL_FORMAT_YCbCr_420_SP_VENUS      0x7FA30C04
#define HAL_PIXEL_FORMAT_YCbCr_420_SP_TILED      0x7FA30C03
#define HAL_PIXEL_FORMAT_YCbCr_420_SP            0x109
#define HAL_PIXEL_FORMAT_YCrCb_420_SP_ADRENO     0x7FA30C01
#define HAL_PIXEL_FORMAT_YCrCb_422_SP            0x10B
#define HAL_PIXEL_FORMAT_R_8                     0x10D
#define HAL_PIXEL_FORMAT_RG_88                   0x10E
#define HAL_PIXEL_FORMAT_YCbCr_444_SP            0x10F
#define HAL_PIXEL_FORMAT_YCrCb_444_SP            0x110
#define HAL_PIXEL_FORMAT_YCrCb_422_I             0x111
#define HAL_PIXEL_FORMAT_BGRX_8888               0x112
#define HAL_PIXEL_FORMAT_NV21_ZSL                0x113
#define HAL_PIXEL_FORMAT_YCrCb_420_SP_VENUS      0x114
#define HAL_PIXEL_FORMAT_BGR_565                 0x115
#define HAL_PIXEL_FORMAT_INTERLACE               0x180

//v4l2_fourcc('Y', 'U', 'Y', 'L'). 24 bpp YUYV 4:2:2 10 bit per component
#define HAL_PIXEL_FORMAT_YCbCr_422_I_10BIT       0x4C595559

//v4l2_fourcc('Y', 'B', 'W', 'C'). 10 bit per component. This compressed
//format reduces the memory access bandwidth
#define HAL_PIXEL_FORMAT_YCbCr_422_I_10BIT_COMPRESSED  0x43574259

// UBWC aligned Venus format
#define HAL_PIXEL_FORMAT_YCbCr_420_SP_VENUS_UBWC 0x7FA30C06

//Khronos ASTC formats
#define HAL_PIXEL_FORMAT_COMPRESSED_RGBA_ASTC_4x4_KHR             0x93B0
#define HAL_PIXEL_FORMAT_COMPRESSED_RGBA_ASTC_5x4_KHR             0x93B1
#define HAL_PIXEL_FORMAT_COMPRESSED_RGBA_ASTC_5x5_KHR             0x93B2
#define HAL_PIXEL_FORMAT_COMPRESSED_RGBA_ASTC_6x5_KHR             0x93B3
#define HAL_PIXEL_FORMAT_COMPRESSED_RGBA_ASTC_6x6_KHR             0x93B4
#define HAL_PIXEL_FORMAT_COMPRESSED_RGBA_ASTC_8x5_KHR             0x93B5
#define HAL_PIXEL_FORMAT_COMPRESSED_RGBA_ASTC_8x6_KHR             0x93B6
#define HAL_PIXEL_FORMAT_COMPRESSED_RGBA_ASTC_8x8_KHR             0x93B7
#define HAL_PIXEL_FORMAT_COMPRESSED_RGBA_ASTC_10x5_KHR            0x93B8
#define HAL_PIXEL_FORMAT_COMPRESSED_RGBA_ASTC_10x6_KHR            0x93B9
#define HAL_PIXEL_FORMAT_COMPRESSED_RGBA_ASTC_10x8_KHR            0x93BA
#define HAL_PIXEL_FORMAT_COMPRESSED_RGBA_ASTC_10x10_KHR           0x93BB
#define HAL_PIXEL_FORMAT_COMPRESSED_RGBA_ASTC_12x10_KHR           0x93BC
#define HAL_PIXEL_FORMAT_COMPRESSED_RGBA_ASTC_12x12_KHR           0x93BD
#define HAL_PIXEL_FORMAT_COMPRESSED_SRGB8_ALPHA8_ASTC_4x4_KHR     0x93D0
#define HAL_PIXEL_FORMAT_COMPRESSED_SRGB8_ALPHA8_ASTC_5x4_KHR     0x93D1
#define HAL_PIXEL_FORMAT_COMPRESSED_SRGB8_ALPHA8_ASTC_5x5_KHR     0x93D2
#define HAL_PIXEL_FORMAT_COMPRESSED_SRGB8_ALPHA8_ASTC_6x5_KHR     0x93D3
#define HAL_PIXEL_FORMAT_COMPRESSED_SRGB8_ALPHA8_ASTC_6x6_KHR     0x93D4
#define HAL_PIXEL_FORMAT_COMPRESSED_SRGB8_ALPHA8_ASTC_8x5_KHR     0x93D5
#define HAL_PIXEL_FORMAT_COMPRESSED_SRGB8_ALPHA8_ASTC_8x6_KHR     0x93D6
#define HAL_PIXEL_FORMAT_COMPRESSED_SRGB8_ALPHA8_ASTC_8x8_KHR     0x93D7
#define HAL_PIXEL_FORMAT_COMPRESSED_SRGB8_ALPHA8_ASTC_10x5_KHR    0x93D8
#define HAL_PIXEL_FORMAT_COMPRESSED_SRGB8_ALPHA8_ASTC_10x6_KHR    0x93D9
#define HAL_PIXEL_FORMAT_COMPRESSED_SRGB8_ALPHA8_ASTC_10x8_KHR    0x93DA
#define HAL_PIXEL_FORMAT_COMPRESSED_SRGB8_ALPHA8_ASTC_10x10_KHR   0x93DB
#define HAL_PIXEL_FORMAT_COMPRESSED_SRGB8_ALPHA8_ASTC_12x10_KHR   0x93DC
#define HAL_PIXEL_FORMAT_COMPRESSED_SRGB8_ALPHA8_ASTC_12x12_KHR   0x93DD

/* possible values for inverse gamma correction */
#define HAL_IGC_NOT_SPECIFIED     0
#define HAL_IGC_s_RGB             1

/* possible formats for 3D content*/
enum {
    HAL_NO_3D                      = 0x0,
    HAL_3D_SIDE_BY_SIDE_L_R        = 0x1,
    HAL_3D_SIDE_BY_SIDE_R_L        = 0x2,
    HAL_3D_TOP_BOTTOM              = 0x4,
    HAL_3D_IN_SIDE_BY_SIDE_L_R     = 0x10000, //unused legacy format
};

enum {
    BUFFER_TYPE_UI = 0,
    BUFFER_TYPE_VIDEO
};

#ifdef __cplusplus
struct private_handle_t : public native_handle {
#else
    struct private_handle_t {
        native_handle_t nativeHandle;
#endif
        enum {
            PRIV_FLAGS_FRAMEBUFFER        = 0x00000001,
            PRIV_FLAGS_USES_ION           = 0x00000008,
            PRIV_FLAGS_USES_ASHMEM        = 0x00000010,
            PRIV_FLAGS_NEEDS_FLUSH        = 0x00000020,
            PRIV_FLAGS_INTERNAL_ONLY      = 0x00000040,
            PRIV_FLAGS_NON_CPU_WRITER     = 0x00000080,
            PRIV_FLAGS_NONCONTIGUOUS_MEM  = 0x00000100,
            PRIV_FLAGS_CACHED             = 0x00000200,
            PRIV_FLAGS_SECURE_BUFFER      = 0x00000400,
            // Display on external only
            PRIV_FLAGS_EXTERNAL_ONLY      = 0x00002000,
            // Set by HWC for protected non secure buffers
            PRIV_FLAGS_PROTECTED_BUFFER   = 0x00004000,
            PRIV_FLAGS_VIDEO_ENCODER      = 0x00010000,
            PRIV_FLAGS_CAMERA_WRITE       = 0x00020000,
            PRIV_FLAGS_CAMERA_READ        = 0x00040000,
            PRIV_FLAGS_HW_COMPOSER        = 0x00080000,
            PRIV_FLAGS_HW_TEXTURE         = 0x00100000,
            PRIV_FLAGS_ITU_R_601          = 0x00200000, //Unused from display
            PRIV_FLAGS_ITU_R_601_FR       = 0x00400000, //Unused from display
            PRIV_FLAGS_ITU_R_709          = 0x00800000, //Unused from display
            PRIV_FLAGS_SECURE_DISPLAY     = 0x01000000,
            // Buffer is rendered in Tile Format
            PRIV_FLAGS_TILE_RENDERED      = 0x02000000,
            // Buffer rendered using CPU/SW renderer
            PRIV_FLAGS_CPU_RENDERED       = 0x04000000,
            // Buffer is allocated with UBWC alignment
            PRIV_FLAGS_UBWC_ALIGNED       = 0x08000000,
            // Buffer allocated will be consumed by SF/HWC
            PRIV_FLAGS_DISP_CONSUMER      = 0x10000000
        };

        // file-descriptors
        int     fd;
        int     fd_metadata;          // fd for the meta-data
        // ints
        int     magic;
        int     flags;
        unsigned int  size;
        unsigned int  offset;
        int     bufferType;
        uint64_t base __attribute__((aligned(8)));
        unsigned int  offset_metadata;
        // The gpu address mapped into the mmu.
        uint64_t gpuaddr __attribute__((aligned(8)));
        int     format;
        int     width;              // specifies aligned width
        int     height;             // specifies aligned height
        int     real_width;
        int     real_height;
        uint64_t base_metadata __attribute__((aligned(8)));

#ifdef __cplusplus
        static const int sNumFds = 2;
        static inline int sNumInts() {
            return ((sizeof(private_handle_t) - sizeof(native_handle_t)) /
                    sizeof(int)) - sNumFds;
        }
        static const int sMagic = 'gmsm';

        private_handle_t(int fd, unsigned int size, int flags, int bufferType,
                         int format, int aligned_width, int aligned_height,
                         int width, int height, int eFd = -1,
                         unsigned int eOffset = 0, uint64_t eBase = 0) :
            fd(fd), fd_metadata(eFd), magic(sMagic),
            flags(flags), size(size), offset(0), bufferType(bufferType),
            base(0), offset_metadata(eOffset), gpuaddr(0),
            format(format), width(aligned_width), height(aligned_height),
            real_width(width), real_height(height), base_metadata(eBase)
        {
            version = (int) sizeof(native_handle);
            numInts = sNumInts();
            numFds = sNumFds;
        }
        ~private_handle_t() {
            magic = 0;
        }

        static int validate(const native_handle* h) {
            const private_handle_t* hnd = (const private_handle_t*)h;
            if (!h || h->version != sizeof(native_handle) ||
                h->numInts != sNumInts() || h->numFds != sNumFds ||
                hnd->magic != sMagic)
            {
                ALOGD("Invalid gralloc handle (at %p): "
                      "ver(%d/%zu) ints(%d/%d) fds(%d/%d)"
                      "magic(%c%c%c%c/%c%c%c%c)",
                      h,
                      h ? h->version : -1, sizeof(native_handle),
                      h ? h->numInts : -1, sNumInts(),
                      h ? h->numFds : -1, sNumFds,
                      hnd ? (((hnd->magic >> 24) & 0xFF)?
                             ((hnd->magic >> 24) & 0xFF) : '-') : '?',
                      hnd ? (((hnd->magic >> 16) & 0xFF)?
                             ((hnd->magic >> 16) & 0xFF) : '-') : '?',
                      hnd ? (((hnd->magic >> 8) & 0xFF)?
                             ((hnd->magic >> 8) & 0xFF) : '-') : '?',
                      hnd ? (((hnd->magic >> 0) & 0xFF)?
                             ((hnd->magic >> 0) & 0xFF) : '-') : '?',
                      (sMagic >> 24) & 0xFF,
                      (sMagic >> 16) & 0xFF,
                      (sMagic >> 8) & 0xFF,
                      (sMagic >> 0) & 0xFF);
                return -EINVAL;
            }
            return 0;
        }

        static private_handle_t* dynamicCast(const native_handle* in) {
            if (validate(in) == 0) {
                return (private_handle_t*) in;
            }
            return NULL;
        }
#endif
    };

#endif /* GRALLOC_PRIV_H_ */
