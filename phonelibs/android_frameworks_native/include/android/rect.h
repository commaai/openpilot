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
 * @addtogroup NativeActivity Native Activity
 * @{
 */

/**
 * @file rect.h
 */

#ifndef ANDROID_RECT_H
#define ANDROID_RECT_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * {@link ARect} is a struct that represents a rectangular window area.
 *
 * It is used with {@link
 * ANativeActivityCallbacks::onContentRectChanged} event callback and
 * ANativeWindow_lock() function.
 */
typedef struct ARect {
#ifdef __cplusplus
    typedef int32_t value_type;
#endif
    /** left position */
    int32_t left;
    /** top position */
    int32_t top;
    /** left position */
    int32_t right;
    /** bottom position */
    int32_t bottom;
} ARect;

#ifdef __cplusplus
};
#endif

#endif // ANDROID_RECT_H

/** @} */
