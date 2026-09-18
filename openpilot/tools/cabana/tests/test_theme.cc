#include <algorithm>
#include <cstdio>

#include "common/tests/native_test.h"
#include "implot.h"
#include "tools/cabana/core/settings.h"
#include "tools/cabana/ui/theme.h"

void test_theme() {
  using namespace cabana::contrast;
  ImGui::CreateContext();
  ImPlot::CreateContext();
  for (int theme : {LIGHT_THEME, DARK_THEME, LIGHT_THEME}) {
    applyTheme(theme);  // Also exercise switching without recreating the UI.
    const auto &p = palette();
    auto check = [](ImVec4 foreground, ImVec4 background, double target) {
      const double value = ratio(fromImVec4(foreground), fromImVec4(background));
      REQUIRE(value >= target);
      return value;
    };
    double text_min = 21, muted_min = 21, control_min = 21;
    for (const auto &background : {p.window, p.surface, p.frame, p.frame_hovered, p.frame_active,
                                  p.button, p.button_hovered, p.button_active, p.header_hovered, p.header_active,
                                  p.tab, p.tab_hovered, p.tab_selected, p.table_header}) {
      text_min = std::min(text_min, check(p.text, background, TEXT));
      muted_min = std::min(muted_min, check(p.text_disabled, background, TEXT));
      control_min = std::min(control_min, check(p.border, background, GRAPHIC));
      check(p.accent, background, GRAPHIC);
    }
    for (const auto &background : {p.window, p.surface, p.frame, p.frame_hovered, p.frame_active}) {
      const auto selection = composite(fromImVec4(ImGui::GetStyleColorVec4(ImGuiCol_TextSelectedBg)), fromImVec4(background));
      REQUIRE(ratio(fromImVec4(p.text), selection) >= TEXT);
    }
    const double selected = check(p.text_selected, p.header, TEXT);
    const double badge = check(p.text_selected, p.badge, TEXT);
    check(p.accent, p.surface, TEXT);  // Links.
    check(p.accent, p.window, TEXT);
    check(p.scrollbar_grab, p.window, GRAPHIC);
    check(p.slider_track, p.window, GRAPHIC);
    for (const auto &color : {CabanaColor(111, 143, 175), CabanaColor(0, 163, 108), CabanaColor(0, 255, 0),
                              CabanaColor(255, 195, 0), CabanaColor(199, 0, 57), CabanaColor(255, 0, 255)}) {
      REQUIRE(ratio(graphicColor(color, p.window), fromImVec4(p.window)) >= GRAPHIC);
      REQUIRE(ratio(foreground(color, fromImVec4(p.text_selected)), fromImVec4(p.text_selected)) >= TEXT);
    }
    double byte_min = 21, bit_min = 21, hover_min = 21, edge_min = 21, plot_min = 21;
    for (int hue = 0; hue < 360; ++hue) {
      for (float saturation : {0.25f, 0.5f, 1.0f}) {
        auto color = CabanaColor::fromHsv(hue / 360.0f, saturation, 1.0f);
        const auto highlight = signalHighlight(color);
        hover_min = std::min(hover_min, ratio(highlight, fromImVec4(p.text_selected)));
        REQUIRE(hover_min >= TEXT);
        const auto outline = signalOutline(color);
        for (const auto &fill : {highlight, fromImVec4(p.surface)}) edge_min = std::min(edge_min, ratio(outline, fill));
        plot_min = std::min(plot_min, ratio(graphicColor(color), fromImVec4(p.surface)));
        REQUIRE(plot_min >= GRAPHIC);
        REQUIRE(ratio(sparklineColor(color), fromImVec4(p.surface)) >= GRAPHIC);
        for (int alpha = 0; alpha < 256; ++alpha) {
          color.a = alpha;
          byte_min = std::min(byte_min, ratio(byteColor(color), fromImVec4(p.text)));
          const auto fill = signalFill(color);
          bit_min = std::min(bit_min, ratio(fill, fromImVec4(p.text)));
          edge_min = std::min(edge_min, ratio(outline, fill));
        }
        REQUIRE(byte_min >= 5.0);
        REQUIRE(bit_min >= TEXT);
        REQUIRE(edge_min >= GRAPHIC);
      }
    }
    printf("%s: text %.4f, muted %.4f, selected %.4f, badge %.4f, controls %.4f, bytes %.4f, bits %.4f, hover %.4f, outlines %.4f, plots %.4f\n",
           theme == LIGHT_THEME ? "light" : "dark", text_min, muted_min, selected, badge, control_min,
           byte_min, bit_min, hover_min, edge_min, plot_min);
  }
  ImPlot::DestroyContext();
  ImGui::DestroyContext();
}

int main() { return run_native_test(test_theme); }
