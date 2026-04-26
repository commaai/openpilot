#pragma once

#include "imgui.h"

#include <cstdint>

enum class AppColor : uint8_t {
  Clear,
  FpsBg,
  FpsBorder,
  FpsText,
  PlotBg,
  PlotBorder,
  PlotLegendBg,
  PlotLegendBorder,
  PlotText,
  PlotMutedText,
  PlotGrid,
  PlotAxisBg,
  PlotAxisBgHovered,
  PlotAxisBgActive,
  PlotSelection,
  PlotCrosshairs,
  PlotCursor,
};

class Style {
 public:
  bool dark_mode() const { return dark_mode_; }
  void set_dark_mode(bool enabled);
  ImVec4 color(AppColor c) const;

 private:
  bool dark_mode_ = false;
};

Style &style();
