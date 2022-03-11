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

#ifndef _LIBINPUT_DISPLAY_VIEWPORT_H
#define _LIBINPUT_DISPLAY_VIEWPORT_H

#include <ui/DisplayInfo.h>
#include <input/Input.h>

namespace android {

/*
 * Describes how coordinates are mapped on a physical display.
 * See com.android.server.display.DisplayViewport.
 */
struct DisplayViewport {
    int32_t displayId; // -1 if invalid
    int32_t orientation;
    int32_t logicalLeft;
    int32_t logicalTop;
    int32_t logicalRight;
    int32_t logicalBottom;
    int32_t physicalLeft;
    int32_t physicalTop;
    int32_t physicalRight;
    int32_t physicalBottom;
    int32_t deviceWidth;
    int32_t deviceHeight;
    String8 uniqueId;

    DisplayViewport() :
            displayId(ADISPLAY_ID_NONE), orientation(DISPLAY_ORIENTATION_0),
            logicalLeft(0), logicalTop(0), logicalRight(0), logicalBottom(0),
            physicalLeft(0), physicalTop(0), physicalRight(0), physicalBottom(0),
            deviceWidth(0), deviceHeight(0) {
    }

    bool operator==(const DisplayViewport& other) const {
        return displayId == other.displayId
                && orientation == other.orientation
                && logicalLeft == other.logicalLeft
                && logicalTop == other.logicalTop
                && logicalRight == other.logicalRight
                && logicalBottom == other.logicalBottom
                && physicalLeft == other.physicalLeft
                && physicalTop == other.physicalTop
                && physicalRight == other.physicalRight
                && physicalBottom == other.physicalBottom
                && deviceWidth == other.deviceWidth
                && deviceHeight == other.deviceHeight
                && uniqueId == other.uniqueId;
    }

    bool operator!=(const DisplayViewport& other) const {
        return !(*this == other);
    }

    inline bool isValid() const {
        return displayId >= 0;
    }

    void setNonDisplayViewport(int32_t width, int32_t height) {
        displayId = ADISPLAY_ID_NONE;
        orientation = DISPLAY_ORIENTATION_0;
        logicalLeft = 0;
        logicalTop = 0;
        logicalRight = width;
        logicalBottom = height;
        physicalLeft = 0;
        physicalTop = 0;
        physicalRight = width;
        physicalBottom = height;
        deviceWidth = width;
        deviceHeight = height;
        uniqueId.clear();
    }
};

/**
 * Describes the different type of viewports supported by input flinger.
 * Keep in sync with values in InputManagerService.java.
 */
enum class ViewportType : int32_t {
    VIEWPORT_INTERNAL = 1,
    VIEWPORT_EXTERNAL = 2,
    VIEWPORT_VIRTUAL = 3,
};

} // namespace android

#endif // _LIBINPUT_DISPLAY_VIEWPORT_H
