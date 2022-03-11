/*
 * Copyright 2017 The Android Open Source Project
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

#pragma once

#include <ui/GraphicTypes.h>
#include <ui/PixelFormat.h>

#include <string>

namespace android {
class Rect;
}

std::string decodeStandard(android_dataspace dataspace);
std::string decodeTransfer(android_dataspace dataspace);
std::string decodeRange(android_dataspace dataspace);
std::string dataspaceDetails(android_dataspace dataspace);
std::string decodeColorMode(android::ui::ColorMode colormode);
std::string decodeColorTransform(android_color_transform colorTransform);
std::string decodePixelFormat(android::PixelFormat format);
std::string decodeRenderIntent(android::ui::RenderIntent renderIntent);
std::string to_string(const android::Rect& rect);
