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

#ifndef ANDROID_PRIVATE_NATIVE_AHARDWARE_BUFFER_HELPERS_H
#define ANDROID_PRIVATE_NATIVE_AHARDWARE_BUFFER_HELPERS_H

/*
 * This file contains utility functions related to AHardwareBuffer, mostly to
 * convert to/from HAL formats.
 *
 * These are PRIVATE methods, so this file can NEVER appear in a public NDK
 * header. They are used by higher level libraries such as core/jni.
 */

#include <stdint.h>

struct AHardwareBuffer;
struct ANativeWindowBuffer;

namespace android {

// whether this AHardwareBuffer format is valid
bool AHardwareBuffer_isValidPixelFormat(uint32_t ahardwarebuffer_format);

// convert AHardwareBuffer format to HAL format (note: this is a no-op)
uint32_t AHardwareBuffer_convertFromPixelFormat(uint32_t format);

// convert HAL format to AHardwareBuffer format (note: this is a no-op)
uint32_t AHardwareBuffer_convertToPixelFormat(uint32_t format);

// convert AHardwareBuffer usage bits to HAL usage bits (note: this is a no-op)
uint64_t AHardwareBuffer_convertFromGrallocUsageBits(uint64_t usage);

// convert HAL usage bits to AHardwareBuffer usage bits  (note: this is a no-op)
uint64_t AHardwareBuffer_convertToGrallocUsageBits(uint64_t usage);

class GraphicBuffer;
const GraphicBuffer* AHardwareBuffer_to_GraphicBuffer(const AHardwareBuffer* buffer);
GraphicBuffer* AHardwareBuffer_to_GraphicBuffer(AHardwareBuffer* buffer);

const ANativeWindowBuffer* AHardwareBuffer_to_ANativeWindowBuffer(const AHardwareBuffer* buffer);
ANativeWindowBuffer* AHardwareBuffer_to_ANativeWindowBuffer(AHardwareBuffer* buffer);

AHardwareBuffer* AHardwareBuffer_from_GraphicBuffer(GraphicBuffer* buffer);
} // namespace android

#endif // ANDROID_PRIVATE_NATIVE_AHARDWARE_BUFFER_HELPERS_H
