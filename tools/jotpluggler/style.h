#pragma once

#include "imgui.h"

#include <cstdint>

enum class AppTheme : uint8_t { Light, Dark };

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

bool app_dark_mode();
ImVec4 app_color(AppColor color);
void apply_app_style(AppTheme theme);
