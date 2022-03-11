/*
 * Copyright (C) 2007 The Android Open Source Project
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

#ifndef ANDROID_UI_REGION_H
#define ANDROID_UI_REGION_H

#include <stdint.h>
#include <sys/types.h>

#include <utils/Vector.h>

#include <ui/Rect.h>
#include <utils/Flattenable.h>

namespace android {
// ---------------------------------------------------------------------------

class String8;

// ---------------------------------------------------------------------------
class Region : public LightFlattenable<Region>
{
public:
    static const Region INVALID_REGION;

                        Region();
                        Region(const Region& rhs);
    explicit            Region(const Rect& rhs);
                        ~Region();

    static  Region      createTJunctionFreeRegion(const Region& r);

        Region& operator = (const Region& rhs);

    inline  bool        isEmpty() const     { return getBounds().isEmpty(); }
    inline  bool        isRect() const      { return mStorage.size() == 1; }

    inline  Rect        getBounds() const   { return mStorage[mStorage.size() - 1]; }
    inline  Rect        bounds() const      { return getBounds(); }

            bool        contains(const Point& point) const;
            bool        contains(int x, int y) const;

            // the region becomes its bounds
            Region&     makeBoundsSelf();

            void        clear();
            void        set(const Rect& r);
            void        set(int32_t w, int32_t h);
            void        set(uint32_t w, uint32_t h);

            Region&     orSelf(const Rect& rhs);
            Region&     xorSelf(const Rect& rhs);
            Region&     andSelf(const Rect& rhs);
            Region&     subtractSelf(const Rect& rhs);

            // boolean operators, applied on this
            Region&     orSelf(const Region& rhs);
            Region&     xorSelf(const Region& rhs);
            Region&     andSelf(const Region& rhs);
            Region&     subtractSelf(const Region& rhs);

            // boolean operators
    const   Region      merge(const Rect& rhs) const;
    const   Region      mergeExclusive(const Rect& rhs) const;
    const   Region      intersect(const Rect& rhs) const;
    const   Region      subtract(const Rect& rhs) const;

            // boolean operators
    const   Region      merge(const Region& rhs) const;
    const   Region      mergeExclusive(const Region& rhs) const;
    const   Region      intersect(const Region& rhs) const;
    const   Region      subtract(const Region& rhs) const;

            // these translate rhs first
            Region&     translateSelf(int dx, int dy);
            Region&     orSelf(const Region& rhs, int dx, int dy);
            Region&     xorSelf(const Region& rhs, int dx, int dy);
            Region&     andSelf(const Region& rhs, int dx, int dy);
            Region&     subtractSelf(const Region& rhs, int dx, int dy);

            // these translate rhs first
    const   Region      translate(int dx, int dy) const;
    const   Region      merge(const Region& rhs, int dx, int dy) const;
    const   Region      mergeExclusive(const Region& rhs, int dx, int dy) const;
    const   Region      intersect(const Region& rhs, int dx, int dy) const;
    const   Region      subtract(const Region& rhs, int dx, int dy) const;

    // convenience operators overloads
    inline  const Region      operator | (const Region& rhs) const;
    inline  const Region      operator ^ (const Region& rhs) const;
    inline  const Region      operator & (const Region& rhs) const;
    inline  const Region      operator - (const Region& rhs) const;
    inline  const Region      operator + (const Point& pt) const;

    inline  Region&     operator |= (const Region& rhs);
    inline  Region&     operator ^= (const Region& rhs);
    inline  Region&     operator &= (const Region& rhs);
    inline  Region&     operator -= (const Region& rhs);
    inline  Region&     operator += (const Point& pt);


    // returns true if the regions share the same underlying storage
    bool isTriviallyEqual(const Region& region) const;


    /* various ways to access the rectangle list */


    // STL-like iterators
    typedef Rect const* const_iterator;
    const_iterator begin() const;
    const_iterator end() const;

    // returns an array of rect which has the same life-time has this
    // Region object.
    Rect const* getArray(size_t* count) const;

    /* no user serviceable parts here... */

            // add a rectangle to the internal list. This rectangle must
            // be sorted in Y and X and must not make the region invalid.
            void        addRectUnchecked(int l, int t, int r, int b);

    inline  bool        isFixedSize() const { return false; }
            size_t      getFlattenedSize() const;
            status_t    flatten(void* buffer, size_t size) const;
            status_t    unflatten(void const* buffer, size_t size);

    void        dump(String8& out, const char* what, uint32_t flags=0) const;
    void        dump(const char* what, uint32_t flags=0) const;

private:
    class rasterizer;
    friend class rasterizer;

    Region& operationSelf(const Rect& r, uint32_t op);
    Region& operationSelf(const Region& r, uint32_t op);
    Region& operationSelf(const Region& r, int dx, int dy, uint32_t op);
    const Region operation(const Rect& rhs, uint32_t op) const;
    const Region operation(const Region& rhs, uint32_t op) const;
    const Region operation(const Region& rhs, int dx, int dy, uint32_t op) const;

    static void boolean_operation(uint32_t op, Region& dst,
            const Region& lhs, const Region& rhs, int dx, int dy);
    static void boolean_operation(uint32_t op, Region& dst,
            const Region& lhs, const Rect& rhs, int dx, int dy);

    static void boolean_operation(uint32_t op, Region& dst,
            const Region& lhs, const Region& rhs);
    static void boolean_operation(uint32_t op, Region& dst,
            const Region& lhs, const Rect& rhs);

    static void translate(Region& reg, int dx, int dy);
    static void translate(Region& dst, const Region& reg, int dx, int dy);

    static bool validate(const Region& reg,
            const char* name, bool silent = false);

    // mStorage is a (manually) sorted array of Rects describing the region
    // with an extra Rect as the last element which is set to the
    // bounds of the region. However, if the region is
    // a simple Rect then mStorage contains only that rect.
    Vector<Rect> mStorage;
};


const Region Region::operator | (const Region& rhs) const {
    return merge(rhs);
}
const Region Region::operator ^ (const Region& rhs) const {
    return mergeExclusive(rhs);
}
const Region Region::operator & (const Region& rhs) const {
    return intersect(rhs);
}
const Region Region::operator - (const Region& rhs) const {
    return subtract(rhs);
}
const Region Region::operator + (const Point& pt) const {
    return translate(pt.x, pt.y);
}


Region& Region::operator |= (const Region& rhs) {
    return orSelf(rhs);
}
Region& Region::operator ^= (const Region& rhs) {
    return xorSelf(rhs);
}
Region& Region::operator &= (const Region& rhs) {
    return andSelf(rhs);
}
Region& Region::operator -= (const Region& rhs) {
    return subtractSelf(rhs);
}
Region& Region::operator += (const Point& pt) {
    return translateSelf(pt.x, pt.y);
}
// ---------------------------------------------------------------------------
}; // namespace android

#endif // ANDROID_UI_REGION_H

