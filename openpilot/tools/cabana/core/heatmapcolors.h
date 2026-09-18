#pragma once

#include "tools/cabana/core/colorcontrast.h"

namespace cabana::heatmap {

using contrast::boundLuminance;
using contrast::composite;
using contrast::luminance;

inline CabanaColor textBackground(const CabanaColor &color, bool white_text) {
  // White: >=4.56:1; black: >=4.6:1. One foreground color per state/theme,
  // including stale messages, rather than changing digits to suit each cell.
  return boundLuminance(color, 0.18, !white_text);
}

inline CabanaColor highlightFill(CabanaColor color) {
  color.a = 255;
  // The lower bound also lets the light-theme signal outline remain distinct
  // from the highlighted fill. Selection and hover share the same text budget.
  return boundLuminance(boundLuminance(color, 0.175, true), 0.18, false);
}

inline CabanaColor bitFill(CabanaColor color, const CabanaColor &base, bool dark, bool defined) {
  auto background = composite(color, base);
  background = composite(dark ? CabanaColor(0, 0, 0, 115)
                              : CabanaColor(255, 255, 255, defined ? 20 : 40), background);
  return textBackground(background, dark);
}

inline CabanaColor byteFill(const CabanaColor &color, const CabanaColor &base, bool dark) {
  const auto hsv = color.hsv();
  auto lavender = CabanaColor::fromHsv(hsv.hue, std::min(hsv.saturation, 0.30f), 0.86f, color.alphaF());
  if (dark) lavender = boundLuminance(lavender, 0.16, false);  // white text >=5:1
  auto background = composite(lavender, base);
  return dark ? background : composite({255, 255, 255, 40}, background);
}

inline CabanaColor outlineColor(const CabanaColor &color, bool dark, bool hovered) {
  // Every neighboring fill is <=0.18 in dark mode. Light fills (including
  // hover/selection) are >=0.17. These bounds guarantee >3:1 on either side
  // of a definition edge, independently of activity and neighboring hues.
  auto edge = dark && hovered ? color.lighter(150) : color;
  edge.a = 255;
  return dark ? boundLuminance(edge, 0.67, true) : boundLuminance(edge, 0.02, false);
}

}  // namespace cabana::heatmap
