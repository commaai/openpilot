#pragma once

#include "imgui.h"
#include "implot.h"

void apply_dark_theme();
void apply_light_theme();
void push_sidebar_style(bool dark_mode);
void push_pane_style(bool dark_mode);
void push_workspace_style(bool dark_mode);
void push_new_tab_button_style(bool dark_mode);
ImU32 new_tab_icon_color(bool dark_mode);
void push_plot_style(bool dark_mode);
