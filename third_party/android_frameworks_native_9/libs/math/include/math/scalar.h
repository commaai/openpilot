/*
 * Copyright (C) 2016 The Android Open Source Project
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

#include <algorithm>
#include <cmath>

namespace android {

template<typename T>
static constexpr T saturate(T v) noexcept {
    return T(std::min(T(1), std::max(T(0), v)));
}

template<typename T>
static constexpr T clamp(T v, T min, T max) noexcept {
    return T(std::min(max, std::max(min, v)));
}

template<typename T>
static constexpr T mix(T x, T y, T a) noexcept {
    return x * (T(1) - a) + y * a;
}

template<typename T>
static constexpr T lerp(T x, T y, T a) noexcept {
    return mix(x, y, a);
}

} // namespace std
