#include "tools/cabana/ui/widgets/scrollabletabbar.h"

#include <algorithm>
#include <cmath>
#include <vector>

#include "imgui_internal.h"

namespace {
float scrollButtonsWidth() {
  const ImGuiStyle &style = ImGui::GetStyle();
  return style.ItemSpacing.x + ImGui::GetFrameHeight() * 2.0f + style.ItemInnerSpacing.x;
}

void drawScrollButtons(ImGuiTabBar *tab_bar) {
  const ImGuiStyle &style = ImGui::GetStyle();
  const float size = ImGui::GetFrameHeight();
  const float max_scroll = std::max(0.0f, tab_bar->WidthAllTabs - tab_bar->BarRect.GetWidth());
  const float start_x = tab_bar->BarRect.Max.x + style.ItemSpacing.x;
  const ImVec2 backup_pos = ImGui::GetCursorScreenPos();

  ImGui::PushItemFlag(ImGuiItemFlags_ButtonRepeat, true);
  for (int i = 0; i < 2; ++i) {
    const bool left = i == 0;
    ImGui::SetCursorScreenPos(ImVec2(start_x + i * (size + style.ItemInnerSpacing.x), tab_bar->BarRect.Min.y));
    ImGui::BeginDisabled(left ? tab_bar->ScrollingTarget <= 0.0f : tab_bar->ScrollingTarget >= max_scroll);
    if (ImGui::Button(left ? "###scroll_left" : "###scroll_right", ImVec2(size, size))) {
      const float step = (left ? -4.0f : 4.0f) * ImGui::GetFontSize();
      tab_bar->ScrollingTarget = std::clamp(tab_bar->ScrollingTarget + step, 0.0f, max_scroll);
      tab_bar->ScrollingAnim = tab_bar->ScrollingTarget;
    }
    // the icon font glyph sits off center in its padded advance, so the chevron is drawn in the rect
    const ImVec2 c((ImGui::GetItemRectMin().x + ImGui::GetItemRectMax().x) * 0.5f,
                   (ImGui::GetItemRectMin().y + ImGui::GetItemRectMax().y) * 0.5f);
    const float h = std::round(ImGui::GetFontSize() * 0.25f);
    const float dx = left ? h * 0.5f : -h * 0.5f;
    ImDrawList *painter = ImGui::GetWindowDrawList();
    painter->PathLineTo(ImVec2(c.x + dx, c.y - h));
    painter->PathLineTo(ImVec2(c.x - dx, c.y));
    painter->PathLineTo(ImVec2(c.x + dx, c.y + h));
    painter->PathStroke(ImGui::GetColorU32(ImGuiCol_Text), ImDrawFlags_None, 1.5f);
    ImGui::EndDisabled();
  }
  ImGui::PopItemFlag();
  ImGui::SetCursorScreenPos(backup_pos);
}

struct ScrollableTabBar { ImGuiTabBar *tab_bar; bool overflowing; };
std::vector<ScrollableTabBar> scrollable_tab_bars;
}  // namespace

bool beginScrollableTabBar(const char *str_id, ImGuiTabBarFlags flags) {
  // the buttons take their room from the bar when the tabs overflowed last frame
  ImGuiWindow *window = ImGui::GetCurrentWindow();
  ImGuiTabBar *prev_tab_bar = ImGui::TabBarFindByID(window->GetID(str_id));
  const bool overflowing = prev_tab_bar && prev_tab_bar->WidthAllTabsIdeal > prev_tab_bar->BarRect.GetWidth() + 1.0f;
  const float backup_work_max_x = window->WorkRect.Max.x;
  if (overflowing) window->WorkRect.Max.x -= scrollButtonsWidth();
  const bool open = ImGui::BeginTabBar(str_id, flags | ImGuiTabBarFlags_FittingPolicyScroll | ImGuiTabBarFlags_NoTabListScrollingButtons);
  window->WorkRect.Max.x = backup_work_max_x;
  if (open) scrollable_tab_bars.push_back({ImGui::GetCurrentTabBar(), overflowing});
  return open;
}

void endScrollableTabBar() {
  ImGui::EndTabBar();
  const ScrollableTabBar bar = scrollable_tab_bars.back();
  scrollable_tab_bars.pop_back();
  if (!bar.overflowing) return;
  drawScrollButtons(bar.tab_bar);

  // the wheel scrolls the tabs while the pointer is over them: a two finger swipe on a touchpad, or a
  // mouse wheel like a window that only scrolls sideways. Owning the wheel keeps the window behind still
  ImGuiTabBar *tab_bar = bar.tab_bar;
  if (ImGui::IsWindowHovered() && ImGui::IsMouseHoveringRect(tab_bar->BarRect.Min, tab_bar->BarRect.Max)) {
    ImGui::SetKeyOwner(ImGuiKey_MouseWheelX, tab_bar->ID);
    ImGui::SetKeyOwner(ImGuiKey_MouseWheelY, tab_bar->ID);
    const ImGuiIO &io = ImGui::GetIO();
    const float wheel = io.MouseWheelH + io.MouseWheel;
    if (wheel != 0.0f) {
      const float max_scroll = std::max(0.0f, tab_bar->WidthAllTabs - tab_bar->BarRect.GetWidth());
      const float step = std::floor(ImGui::GetFontSize() * 2.0f);
      tab_bar->ScrollingTarget = std::clamp(tab_bar->ScrollingTarget - wheel * step, 0.0f, max_scroll);
      tab_bar->ScrollingAnim = tab_bar->ScrollingTarget;
    }
  }
}

