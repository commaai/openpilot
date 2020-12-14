#pragma once
#include "ui.hpp"


void ui_draw_vision_alert(UIState *s, cereal::ControlsState::AlertSize va_size, int va_color,
                          const char* va_text1, const char* va_text2);
void ui_draw(UIState *s);
void ui_draw_image(NVGcontext *vg, float x, float y, float w, float h, int image, float alpha);
void ui_draw_rect(NVGcontext *vg, float x, float y, float w, float h, NVGcolor color, float r = 0, int width = 0);
void ui_draw_rect(NVGcontext *vg, float x, float y, float w, float h, NVGpaint &paint, float r = 0);
void ui_nvg_init(UIState *s);
