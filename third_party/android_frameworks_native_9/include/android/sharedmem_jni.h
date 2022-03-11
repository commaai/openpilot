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

/**
 * @addtogroup Memory
 * @{
 */

/**
 * @file sharedmem_jni.h
 */

#ifndef ANDROID_SHARED_MEMORY_JNI_H
#define ANDROID_SHARED_MEMORY_JNI_H

#include <jni.h>
#include <android/sharedmem.h>
#include <stddef.h>

/******************************************************************
 *
 * IMPORTANT NOTICE:
 *
 *   This file is part of Android's set of stable system headers
 *   exposed by the Android NDK (Native Development Kit).
 *
 *   Third-party source AND binary code relies on the definitions
 *   here to be FROZEN ON ALL UPCOMING PLATFORM RELEASES.
 *
 *   - DO NOT MODIFY ENUMS (EXCEPT IF YOU ADD NEW 32-BIT VALUES)
 *   - DO NOT MODIFY CONSTANTS OR FUNCTIONAL MACROS
 *   - DO NOT CHANGE THE SIGNATURE OF FUNCTIONS IN ANY WAY
 *   - DO NOT CHANGE THE LAYOUT OR SIZE OF STRUCTURES
 */

/**
 * Structures and functions for a shared memory buffer that can be shared across process.
 */

#ifdef __cplusplus
extern "C" {
#endif

#if __ANDROID_API__ >= __ANDROID_API_O_MR1__

/**
 * Returns a dup'd FD from the given Java android.os.SharedMemory object. The returned file
 * descriptor has all the same properties & capabilities as the FD returned from
 * ASharedMemory_create(), however the protection flags will be the same as those of the
 * android.os.SharedMemory object.
 *
 * Use close() to release the shared memory region.
 *
 * \param env The JNIEnv* pointer
 * \param sharedMemory The Java android.os.SharedMemory object
 * \return file descriptor that denotes the shared memory; -1 if the shared memory object is
 *      already closed, if the JNIEnv or jobject is NULL, or if there are too many open file
 *      descriptors (errno=EMFILE)
 */
int ASharedMemory_dupFromJava(JNIEnv* env, jobject sharedMemory);

#endif

#ifdef __cplusplus
};
#endif

#endif // ANDROID_SHARED_MEMORY_JNI_H

/** @} */
