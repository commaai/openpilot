#pragma once

#include "tools/cabana/core/color.h"

namespace cabana::contrast {

// WCAG relative luminance and contrast use opaque, composited sRGB colors.
constexpr double TEXT = 4.5;
constexpr double GRAPHIC = 3.0;

inline double linearChannel(double value) {
  value /= 255.0;
  return value <= 0.04045 ? value / 12.92 : std::pow((value + 0.055) / 1.055, 2.4);
}

inline double luminance(const CabanaColor &color) {
  return 0.2126 * linearChannel(color.r) + 0.7152 * linearChannel(color.g) + 0.0722 * linearChannel(color.b);
}

// Move toward white or black in linear RGB. Directed rounding keeps the bound
// after conversion back to the 8-bit colors used by the renderer.
inline CabanaColor boundLuminance(const CabanaColor &color, double bound, bool minimum) {
  const double current = luminance(color);
  if (minimum ? current >= bound : current <= bound) return color;
  auto channel = [&](uint8_t value) {
    double c = linearChannel(value);
    c = minimum ? c + (1.0 - c) * (bound - current) / (1.0 - current) : c * bound / current;
    const double srgb = c <= 0.0031308 ? 12.92 * c : 1.055 * std::pow(c, 1.0 / 2.4) - 0.055;
    const double scaled = std::clamp(srgb * 255.0, 0.0, 255.0);
    return static_cast<uint8_t>(minimum ? std::ceil(scaled) : std::floor(scaled));
  };
  return {channel(color.r), channel(color.g), channel(color.b), color.a};
}

inline CabanaColor composite(const CabanaColor &color, const CabanaColor &base) {
  auto channel = [&](int foreground, int background) {
    return static_cast<uint8_t>(std::lround((foreground * color.a + background * (255 - color.a)) / 255.0));
  };
  return {channel(color.r, base.r), channel(color.g, base.g), channel(color.b, base.b)};
}

inline double ratio(const CabanaColor &a, const CabanaColor &b) {
  const double l = luminance(a), r = luminance(b);
  return (std::max(l, r) + 0.05) / (std::min(l, r) + 0.05);
}

inline CabanaColor foreground(CabanaColor color, const CabanaColor &background, double target = TEXT) {
  color.a = 255;
  if (ratio(color, background) >= target) return color;
  const double base = luminance(background);
  const bool lighter = base < 0.179;
  const double bound = lighter ? target * (base + 0.05) - 0.05 : (base + 0.05) / target - 0.05;
  return boundLuminance(color, std::clamp(bound, 0.0, 1.0), lighter);
}

}  // namespace cabana::contrast
