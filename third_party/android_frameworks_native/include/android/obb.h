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

/**
 * @addtogroup Storage
 * @{
 */

/**
 * @file obb.h
 */

#ifndef ANDROID_OBB_H
#define ANDROID_OBB_H

#include <sys/types.h>

#ifdef __cplusplus
extern "C" {
#endif

struct AObbInfo;
/** {@link AObbInfo} is an opaque type representing information for obb storage. */
typedef struct AObbInfo AObbInfo;

/** Flag for an obb file, returned by AObbInfo_getFlags(). */
enum {
    /** overlay */
    AOBBINFO_OVERLAY = 0x0001,
};

/**
 * Scan an OBB and get information about it.
 */
AObbInfo* AObbScanner_getObbInfo(const char* filename);

/**
 * Destroy the AObbInfo object. You must call this when finished with the object.
 */
void AObbInfo_delete(AObbInfo* obbInfo);

/**
 * Get the package name for the OBB.
 */
const char* AObbInfo_getPackageName(AObbInfo* obbInfo);

/**
 * Get the version of an OBB file.
 */
int32_t AObbInfo_getVersion(AObbInfo* obbInfo);

/**
 * Get the flags of an OBB file.
 */
int32_t AObbInfo_getFlags(AObbInfo* obbInfo);

#ifdef __cplusplus
};
#endif

#endif      // ANDROID_OBB_H

/** @} */
