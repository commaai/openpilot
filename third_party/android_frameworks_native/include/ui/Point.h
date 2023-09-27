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

#ifndef ANDROID_UI_POINT
#define ANDROID_UI_POINT

#include <utils/Flattenable.h>
#include <utils/TypeHelpers.h>

namespace android {

class Point : public LightFlattenablePod<Point>
{
public:
    int x;
    int y;

    // we don't provide copy-ctor and operator= on purpose
    // because we want the compiler generated versions

    // Default constructor doesn't initialize the Point
    inline Point() {
    }
    inline Point(int x, int y) : x(x), y(y) {
    }

    inline bool operator == (const Point& rhs) const {
        return (x == rhs.x) && (y == rhs.y);
    }
    inline bool operator != (const Point& rhs) const {
        return !operator == (rhs);
    }

    inline bool isOrigin() const {
        return !(x|y);
    }

    // operator < defines an order which allows to use points in sorted
    // vectors.
    bool operator < (const Point& rhs) const {
        return y<rhs.y || (y==rhs.y && x<rhs.x);
    }

    inline Point& operator - () {
        x = -x;
        y = -y;
        return *this;
    }
    
    inline Point& operator += (const Point& rhs) {
        x += rhs.x;
        y += rhs.y;
        return *this;
    }
    inline Point& operator -= (const Point& rhs) {
        x -= rhs.x;
        y -= rhs.y;
        return *this;
    }
    
    const Point operator + (const Point& rhs) const {
        const Point result(x+rhs.x, y+rhs.y);
        return result;
    }
    const Point operator - (const Point& rhs) const {
        const Point result(x-rhs.x, y-rhs.y);
        return result;
    }    
};

ANDROID_BASIC_TYPES_TRAITS(Point)

}; // namespace android

#endif // ANDROID_UI_POINT
