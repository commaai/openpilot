/*
 * Copyright (C) 2006 The Android Open Source Project
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

#ifndef ANDROID_UI_RECT
#define ANDROID_UI_RECT

#include <utils/Flattenable.h>
#include <utils/Log.h>
#include <utils/TypeHelpers.h>
#include <ui/Point.h>

#include <android/rect.h>

namespace android {

class Rect : public ARect, public LightFlattenablePod<Rect>
{
public:
    typedef ARect::value_type value_type;

    static const Rect INVALID_RECT;

    // we don't provide copy-ctor and operator= on purpose
    // because we want the compiler generated versions

    inline Rect() {
      left = right = top = bottom = 0;
    }

    inline Rect(int32_t w, int32_t h) {
        left = top = 0;
        right = w;
        bottom = h;
    }

    inline Rect(uint32_t w, uint32_t h) {
        if (w > INT32_MAX) {
            ALOG(LOG_WARN, "Rect",
                    "Width %u too large for Rect class, clamping", w);
            w = INT32_MAX;
        }
        if (h > INT32_MAX) {
            ALOG(LOG_WARN, "Rect",
                    "Height %u too large for Rect class, clamping", h);
            h = INT32_MAX;
        }
        left = top = 0;
        right = w;
        bottom = h;
    }

    inline Rect(int32_t l, int32_t t, int32_t r, int32_t b) {
        left = l;
        top = t;
        right = r;
        bottom = b;
    }

    inline Rect(const Point& lt, const Point& rb) {
        left = lt.x;
        top = lt.y;
        right = rb.x;
        bottom = rb.y;
    }

    void makeInvalid();

    inline void clear() {
        left = top = right = bottom = 0;
    }

    // a valid rectangle has a non negative width and height
    inline bool isValid() const {
        return (getWidth() >= 0) && (getHeight() >= 0);
    }

    // an empty rect has a zero width or height, or is invalid
    inline bool isEmpty() const {
        return (getWidth() <= 0) || (getHeight() <= 0);
    }

    // rectangle's width
    inline int32_t getWidth() const {
        return right - left;
    }

    // rectangle's height
    inline int32_t getHeight() const {
        return bottom - top;
    }

    inline Rect getBounds() const {
        return Rect(right - left, bottom - top);
    }

    void setLeftTop(const Point& lt) {
        left = lt.x;
        top = lt.y;
    }

    void setRightBottom(const Point& rb) {
        right = rb.x;
        bottom = rb.y;
    }
    
    // the following 4 functions return the 4 corners of the rect as Point
    Point leftTop() const {
        return Point(left, top);
    }
    Point rightBottom() const {
        return Point(right, bottom);
    }
    Point rightTop() const {
        return Point(right, top);
    }
    Point leftBottom() const {
        return Point(left, bottom);
    }

    // comparisons
    inline bool operator == (const Rect& rhs) const {
        return (left == rhs.left) && (top == rhs.top) &&
               (right == rhs.right) && (bottom == rhs.bottom);
    }

    inline bool operator != (const Rect& rhs) const {
        return !operator == (rhs);
    }

    // operator < defines an order which allows to use rectangles in sorted
    // vectors.
    bool operator < (const Rect& rhs) const;

    const Rect operator + (const Point& rhs) const;
    const Rect operator - (const Point& rhs) const;

    Rect& operator += (const Point& rhs) {
        return offsetBy(rhs.x, rhs.y);
    }
    Rect& operator -= (const Point& rhs) {
        return offsetBy(-rhs.x, -rhs.y);
    }

    Rect& offsetToOrigin() {
        right -= left;
        bottom -= top;
        left = top = 0;
        return *this;
    }
    Rect& offsetTo(const Point& p) {
        return offsetTo(p.x, p.y);
    }
    Rect& offsetBy(const Point& dp) {
        return offsetBy(dp.x, dp.y);
    }

    Rect& offsetTo(int32_t x, int32_t y);
    Rect& offsetBy(int32_t x, int32_t y);

    bool intersect(const Rect& with, Rect* result) const;

    // Create a new Rect by transforming this one using a graphics HAL
    // transform.  This rectangle is defined in a coordinate space starting at
    // the origin and extending to (width, height).  If the transform includes
    // a ROT90 then the output rectangle is defined in a space extending to
    // (height, width).  Otherwise the output rectangle is in the same space as
    // the input.
    Rect transform(uint32_t xform, int32_t width, int32_t height) const;

    // this calculates (Region(*this) - exclude).bounds() efficiently
    Rect reduce(const Rect& exclude) const;


    // for backward compatibility
    inline int32_t width() const { return getWidth(); }
    inline int32_t height() const { return getHeight(); }
    inline void set(const Rect& rhs) { operator = (rhs); }
};

ANDROID_BASIC_TYPES_TRAITS(Rect)

}; // namespace android

#endif // ANDROID_UI_RECT
