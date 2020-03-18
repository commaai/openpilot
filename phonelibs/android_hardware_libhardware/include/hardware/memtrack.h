/*
 * Copyright (C) 2013 The Android Open Source Project
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

#ifndef ANDROID_INCLUDE_HARDWARE_MEMTRACK_H
#define ANDROID_INCLUDE_HARDWARE_MEMTRACK_H

#include <stdint.h>
#include <sys/cdefs.h>
#include <sys/types.h>

#include <hardware/hardware.h>

__BEGIN_DECLS

#define MEMTRACK_MODULE_API_VERSION_0_1  HARDWARE_MODULE_API_VERSION(0, 1)

/**
 * The id of this module
 */
#define MEMTRACK_HARDWARE_MODULE_ID "memtrack"

/*
 * The Memory Tracker HAL is designed to return information about device-specific
 * memory usage.  The primary goal is to be able to track memory that is not
 * trackable in any other way, for example texture memory that is allocated by
 * a process, but not mapped in to that process' address space.
 * A secondary goal is to be able to categorize memory used by a process into
 * GL, graphics, etc.  All memory sizes should be in real memory usage,
 * accounting for stride, bit depth, rounding up to page size, etc.
 *
 * A process collecting memory statistics will call getMemory for each
 * combination of pid and memory type.  For each memory type that it recognizes
 * the HAL should fill out an array of memtrack_record structures breaking
 * down the statistics of that memory type as much as possible.  For example,
 * getMemory(<pid>, MEMTRACK_TYPE_GL) might return:
 * { { 4096,  ACCOUNTED | PRIVATE | SYSTEM },
 *   { 40960, UNACCOUNTED | PRIVATE | SYSTEM },
 *   { 8192,  ACCOUNTED | PRIVATE | DEDICATED },
 *   { 8192,  UNACCOUNTED | PRIVATE | DEDICATED } }
 * If the HAL could not differentiate between SYSTEM and DEDICATED memory, it
 * could return:
 * { { 12288,  ACCOUNTED | PRIVATE },
 *   { 49152,  UNACCOUNTED | PRIVATE } }
 *
 * Memory should not overlap between types.  For example, a graphics buffer
 * that has been mapped into the GPU as a surface should show up when
 * MEMTRACK_TYPE_GRAPHICS is requested, and not when MEMTRACK_TYPE_GL
 * is requested.
 */

enum memtrack_type {
    MEMTRACK_TYPE_OTHER = 0,
    MEMTRACK_TYPE_GL = 1,
    MEMTRACK_TYPE_GRAPHICS = 2,
    MEMTRACK_TYPE_MULTIMEDIA = 3,
    MEMTRACK_TYPE_CAMERA = 4,
    MEMTRACK_NUM_TYPES,
};

struct memtrack_record {
    size_t size_in_bytes;
    unsigned int flags;
};

/**
 * Flags to differentiate memory that can already be accounted for in
 * /proc/<pid>/smaps,
 * (Shared_Clean + Shared_Dirty + Private_Clean + Private_Dirty = Size).
 * In general, memory mapped in to a userspace process is accounted unless
 * it was mapped with remap_pfn_range.
 * Exactly one of these should be set.
 */
#define MEMTRACK_FLAG_SMAPS_ACCOUNTED   (1 << 1)
#define MEMTRACK_FLAG_SMAPS_UNACCOUNTED (1 << 2)

/**
 * Flags to differentiate memory shared across multiple processes vs. memory
 * used by a single process.  Only zero or one of these may be set in a record.
 * If none are set, record is assumed to count shared + private memory.
 */
#define MEMTRACK_FLAG_SHARED      (1 << 3)
#define MEMTRACK_FLAG_SHARED_PSS  (1 << 4) /* shared / num_procesess */
#define MEMTRACK_FLAG_PRIVATE     (1 << 5)

/**
 * Flags to differentiate memory taken from the kernel's allocation pool vs.
 * memory that is dedicated to non-kernel allocations, for example a carveout
 * or separate video memory.  Only zero or one of these may be set in a record.
 * If none are set, record is assumed to count system + dedicated memory.
 */
#define MEMTRACK_FLAG_SYSTEM     (1 << 6)
#define MEMTRACK_FLAG_DEDICATED  (1 << 7)

/**
 * Flags to differentiate memory accessible by the CPU in non-secure mode vs.
 * memory that is protected.  Only zero or one of these may be set in a record.
 * If none are set, record is assumed to count secure + nonsecure memory.
 */
#define MEMTRACK_FLAG_NONSECURE  (1 << 8)
#define MEMTRACK_FLAG_SECURE     (1 << 9)

/**
 * Every hardware module must have a data structure named HAL_MODULE_INFO_SYM
 * and the fields of this data structure must begin with hw_module_t
 * followed by module specific information.
 */
typedef struct memtrack_module {
    struct hw_module_t common;

    /**
     * (*init)() performs memtrack management setup actions and is called
     * once before any calls to getMemory().
     * Returns 0 on success, -errno on error.
     */
    int (*init)(const struct memtrack_module *module);

    /**
     * (*getMemory)() expects an array of record objects and populates up to
     * *num_record structures with the sizes of memory plus associated flags for
     * that memory.  It also updates *num_records with the total number of
     * records it could return if *num_records was large enough when passed in.
     * Returning records with size 0 is expected, the number of records should
     * not vary between calls to getMemory for the same memory type, even
     * for different pids.
     *
     * The caller will often call getMemory for a type and pid with
     * *num_records == 0 to determine how many records to allocate room for,
     * this case should be a fast-path in the HAL, returning a constant and
     * not querying any kernel files.  If *num_records passed in is 0,
     * then records may be NULL.
     *
     * This function must be thread-safe, it may get called from multiple
     * threads at the same time.
     *
     * Returns 0 on success, -ENODEV if the type is not supported, -errno
     * on other errors.
     */
    int (*getMemory)(const struct memtrack_module *module,
                     pid_t pid,
                     int type,
                     struct memtrack_record *records,
                     size_t *num_records);
} memtrack_module_t;

__END_DECLS

#endif  // ANDROID_INCLUDE_HARDWARE_MEMTRACK_H
