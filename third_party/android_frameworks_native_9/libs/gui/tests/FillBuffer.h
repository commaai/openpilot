/*
 * Copyright 2013 The Android Open Source Project
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

#ifndef ANDROID_FILL_BUFFER_H
#define ANDROID_FILL_BUFFER_H

#include <system/window.h>
#include <utils/StrongPointer.h>

namespace android {

// Fill a YV12 buffer with a multi-colored checkerboard pattern
void fillYV12Buffer(uint8_t* buf, int w, int h, int stride);

// Fill a YV12 buffer with red outside a given rectangle and green inside it.
void fillYV12BufferRect(uint8_t* buf, int w, int h, int stride,
        const android_native_rect_t& rect);

void fillRGBA8Buffer(uint8_t* buf, int w, int h, int stride);

// Produce a single RGBA8 frame by filling a buffer with a checkerboard pattern
// using the CPU.  This assumes that the ANativeWindow is already configured to
// allow this to be done (e.g. the format is set to RGBA8).
//
// Calls to this function should be wrapped in an ASSERT_NO_FATAL_FAILURE().
void produceOneRGBA8Frame(const sp<ANativeWindow>& anw);

} // namespace android

#endif
