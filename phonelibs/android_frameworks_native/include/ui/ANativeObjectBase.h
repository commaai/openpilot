/*
 * Copyright (C) 2009 The Android Open Source Project
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

#ifndef ANDROID_ANDROID_NATIVES_H
#define ANDROID_ANDROID_NATIVES_H

#include <sys/types.h>
#include <string.h>

#include <hardware/gralloc.h>
#include <system/window.h>

// ---------------------------------------------------------------------------

/* FIXME: this is legacy for pixmaps */
typedef struct egl_native_pixmap_t
{
    int32_t     version;    /* must be 32 */
    int32_t     width;
    int32_t     height;
    int32_t     stride;
    uint8_t*    data;
    uint8_t     format;
    uint8_t     rfu[3];
    union {
        uint32_t    compressedFormat;
        int32_t     vstride;
    };
    int32_t     reserved;
} egl_native_pixmap_t;

/*****************************************************************************/

#ifdef __cplusplus

#include <utils/RefBase.h>

namespace android {

/*
 * This helper class turns a ANativeXXX object type into a C++
 * reference-counted object; with proper type conversions.
 */
template <typename NATIVE_TYPE, typename TYPE, typename REF>
class ANativeObjectBase : public NATIVE_TYPE, public REF
{
public:
    // Disambiguate between the incStrong in REF and NATIVE_TYPE
    void incStrong(const void* id) const {
        REF::incStrong(id);
    }
    void decStrong(const void* id) const {
        REF::decStrong(id);
    }

protected:
    typedef ANativeObjectBase<NATIVE_TYPE, TYPE, REF> BASE;
    ANativeObjectBase() : NATIVE_TYPE(), REF() {
        NATIVE_TYPE::common.incRef = incRef;
        NATIVE_TYPE::common.decRef = decRef;
    }
    static inline TYPE* getSelf(NATIVE_TYPE* self) {
        return static_cast<TYPE*>(self);
    }
    static inline TYPE const* getSelf(NATIVE_TYPE const* self) {
        return static_cast<TYPE const *>(self);
    }
    static inline TYPE* getSelf(android_native_base_t* base) {
        return getSelf(reinterpret_cast<NATIVE_TYPE*>(base));
    }
    static inline TYPE const * getSelf(android_native_base_t const* base) {
        return getSelf(reinterpret_cast<NATIVE_TYPE const*>(base));
    }
    static void incRef(android_native_base_t* base) {
        ANativeObjectBase* self = getSelf(base);
        self->incStrong(self);
    }
    static void decRef(android_native_base_t* base) {
        ANativeObjectBase* self = getSelf(base);
        self->decStrong(self);
    }
};

} // namespace android
#endif // __cplusplus

/*****************************************************************************/

#endif /* ANDROID_ANDROID_NATIVES_H */
