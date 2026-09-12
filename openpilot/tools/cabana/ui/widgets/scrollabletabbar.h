#pragma once

#include "imgui.h"

// a tab bar that scrolls with a pair of chevron buttons at its right end when the tabs overflow, in place of
// imgui's small arrows. Use like BeginTabBar/EndTabBar, the fitting policy is always scroll
bool beginScrollableTabBar(const char *str_id, ImGuiTabBarFlags flags = 0);
void endScrollableTabBar();
