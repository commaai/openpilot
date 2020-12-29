#pragma once
#include "ui.hpp"

void ui_draw(UIState *s);
void ui_draw_image(NVGcontext *vg, float x, float y, float w, float h, int image, float alpha);
void ui_draw_rect(NVGcontext *vg, float x, float y, float w, float h, NVGcolor color, int width = 0, float r = 0);
template <class T>
void ui_fill_rect(NVGcontext *vg, float x, float y, float w, float h, const T &fill, float r = 0) {
  nvgBeginPath(vg);
  r > 0 ? nvgRoundedRect(vg, x, y, w, h, r) : nvgRect(vg, x, y, w, h);
  if constexpr (std::is_same<T, NVGcolor>::value) {
    nvgFillColor(vg, fill);
  } else {
    nvgFillPaint(vg, fill);
  }
  nvgFill(vg);
}
void ui_nvg_init(UIState *s);
