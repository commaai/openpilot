#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>

struct CabanaColor {
  uint8_t r = 0;
  uint8_t g = 0;
  uint8_t b = 0;
  uint8_t a = 255;

  constexpr CabanaColor() = default;
  constexpr CabanaColor(uint8_t red, uint8_t green, uint8_t blue, uint8_t alpha = 255)
      : r(red), g(green), b(blue), a(alpha) {}

  static CabanaColor fromHsv(float hue, float saturation, float value, float alpha = 1.0f) {
    const float h = hue - std::floor(hue);
    const float c = value * saturation;
    const float x = c * (1.0f - std::fabs(std::fmod(h * 6.0f, 2.0f) - 1.0f));
    const float m = value - c;
    float red = 0, green = 0, blue = 0;
    switch (static_cast<int>(h * 6.0f) % 6) {
      case 0: red = c; green = x; break;
      case 1: red = x; green = c; break;
      case 2: green = c; blue = x; break;
      case 3: green = x; blue = c; break;
      case 4: red = x; blue = c; break;
      default: red = c; blue = x; break;
    }
    auto channel = [m](float v) { return static_cast<uint8_t>(std::clamp((v + m) * 255.0f, 0.0f, 255.0f) + 0.5f); };
    return {channel(red), channel(green), channel(blue),
            static_cast<uint8_t>(std::clamp(alpha * 255.0f, 0.0f, 255.0f) + 0.5f)};
  }

  CabanaColor darker(int factor = 200) const {
    if (factor <= 0) return *this;
    if (factor < 100) return lighter(10000 / factor);
    auto [hue, saturation, value] = hsv();
    return fromHsv(hue, saturation, value * 100.0f / factor, a / 255.0f);
  }

  CabanaColor lighter(int factor = 150) const {
    if (factor <= 0) return *this;
    if (factor < 100) return darker(10000 / factor);
    auto [hue, saturation, value] = hsv();
    const float scaled_value = value * factor / 100.0f;
    if (scaled_value > 1.0f) saturation = std::max(0.0f, saturation - (scaled_value - 1.0f));
    return fromHsv(hue, saturation, std::min(1.0f, scaled_value), a / 255.0f);
  }

  constexpr int red() const { return r; }
  constexpr int green() const { return g; }
  constexpr int blue() const { return b; }
  constexpr int alpha() const { return a; }
  float alphaF() const { return a / 255.0f; }
  void setAlphaF(float alpha) { a = static_cast<uint8_t>(std::clamp(alpha * 255.0f, 0.0f, 255.0f) + 0.5f); }

  constexpr bool operator==(const CabanaColor &other) const {
    return r == other.r && g == other.g && b == other.b && a == other.a;
  }

private:
  struct Hsv { float hue; float saturation; float value; };
  Hsv hsv() const {
    const float red = r / 255.0f, green = g / 255.0f, blue = b / 255.0f;
    const float maximum = std::max({red, green, blue});
    const float minimum = std::min({red, green, blue});
    const float delta = maximum - minimum;
    float hue = 0;
    if (delta > 0) {
      if (maximum == red) hue = std::fmod((green - blue) / delta, 6.0f) / 6.0f;
      else if (maximum == green) hue = ((blue - red) / delta + 2.0f) / 6.0f;
      else hue = ((red - green) / delta + 4.0f) / 6.0f;
      if (hue < 0) hue += 1.0f;
    }
    return {hue, maximum == 0 ? 0 : delta / maximum, maximum};
  }
};
