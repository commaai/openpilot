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

#ifndef _LIBINPUT_VIRTUAL_KEY_MAP_H
#define _LIBINPUT_VIRTUAL_KEY_MAP_H

#include <stdint.h>

#include <input/Input.h>
#include <utils/Errors.h>
#include <utils/KeyedVector.h>
#include <utils/Tokenizer.h>
#include <utils/String8.h>
#include <utils/Unicode.h>

namespace android {

/* Describes a virtual key. */
struct VirtualKeyDefinition {
    int32_t scanCode;

    // configured position data, specified in display coords
    int32_t centerX;
    int32_t centerY;
    int32_t width;
    int32_t height;
};


/**
 * Describes a collection of virtual keys on a touch screen in terms of
 * virtual scan codes and hit rectangles.
 *
 * This object is immutable after it has been loaded.
 */
class VirtualKeyMap {
public:
    ~VirtualKeyMap();

    static status_t load(const String8& filename, VirtualKeyMap** outMap);

    inline const Vector<VirtualKeyDefinition>& getVirtualKeys() const {
        return mVirtualKeys;
    }

private:
    class Parser {
        VirtualKeyMap* mMap;
        Tokenizer* mTokenizer;

    public:
        Parser(VirtualKeyMap* map, Tokenizer* tokenizer);
        ~Parser();
        status_t parse();

    private:
        bool consumeFieldDelimiterAndSkipWhitespace();
        bool parseNextIntField(int32_t* outValue);
    };

    Vector<VirtualKeyDefinition> mVirtualKeys;

    VirtualKeyMap();
};

} // namespace android

#endif // _LIBINPUT_KEY_CHARACTER_MAP_H
