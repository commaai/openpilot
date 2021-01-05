#pragma once
#include "ui.hpp"

bool car_space_to_full_frame(const UIState *s, float in_x, float in_y, float in_z, vertex_data *out, float margin=0.0);
void ui_draw(UIState *s);
void ui_draw_image(NVGcontext *vg, const Rect &r, int image, float alpha);
void ui_draw_rect(NVGcontext *vg, const Rect &r, NVGcolor color, float radius = 0, int width = 0);
void ui_draw_rect(NVGcontext *vg, const Rect &r, NVGpaint &paint, float radius = 0);
void ui_nvg_init(UIState *s);
