#pragma once

#include "selfdrive/ui/ui.h"

void ui_draw(UIState *s, int w, int h);
void ui_nvg_init(UIState *s);
void ui_resize(UIState *s, int width, int height);
void ui_update_params(UIState *s);
