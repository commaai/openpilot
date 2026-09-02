#include "tools/cabana/ui/widgets/tabbar.h"

#include <algorithm>
#include <utility>

#include "tools/cabana/ui/widgets/scrollabletabbar.h"

int TabBar::addTab(const std::string &text) {
  tabs_.push_back({text, 0, next_id_++});
  int index = count() - 1;
  if (current_index_ == -1) {  // the first tab is current
    current_index_ = index;
    select_current_ = true;
    currentChanged(index);
  }
  return index;
}

void TabBar::setCurrentIndex(int index) {
  if (index == current_index_ || index < -1 || index >= count()) return;
  current_index_ = index;
  select_current_ = true;
  currentChanged(index);
}

int TabBar::tabAt(const ImVec2 &pos) const {
  for (int i = 0; i < count(); ++i) {
    if (tabs_[i].rect.Contains(pos)) return i;
  }
  return -1;
}

void TabBar::removeTab(int index) {
  tabs_.erase(tabs_.begin() + index);
  if (index == current_index_) {
    // select the tab that moved into this index, else the one to the left
    current_index_ = count() ? std::min(index, count() - 1) : -1;
    select_current_ = true;
    currentChanged(current_index_);
  } else if (index < current_index_) {
    --current_index_;
  }
}

void TabBar::moveTab(int from, int to) {
  if (from == to || from < 0 || from >= count() || to < 0 || to >= count()) return;
  const int current_id = current_index_ >= 0 ? tabs_[current_index_].id : -1;
  Tab tab = std::move(tabs_[from]);
  tabs_.erase(tabs_.begin() + from);
  tabs_.insert(tabs_.begin() + to, std::move(tab));
  for (int i = 0; i < count(); ++i) {
    if (tabs_[i].id == current_id) current_index_ = i;
  }
  select_current_ = true;  // imgui orders the tabs as submitted only when a tab is (re)selected
}

void TabBar::draw() {
  if (auto_hide_ && count() < 2) return;  // auto hidden with fewer than two tabs
  ImGui::PushID(this);
  // no default tooltip, the tabs carry their own
  if (!(scroll_buttons_ ? beginScrollableTabBar("##tabbar", ImGuiTabBarFlags_NoTooltip) : ImGui::BeginTabBar("##tabbar", ImGuiTabBarFlags_NoTooltip))) {
    ImGui::PopID();
    return;
  }
  // every tab gets a close button, not only the hovered/selected one
  ImGuiStyle &style = ImGui::GetStyle();
  const float close_button_min_width = tabs_closable_ ? std::exchange(style.TabCloseButtonMinWidthUnselected, -1.0f) : 0.0f;
  // setCurrentIndex requests are applied on the next frame
  const bool select_current = std::exchange(select_current_, false);
  int close_index = -1;
  for (int i = 0; i < count(); ++i) {
    bool open = true;
    const std::string label = tabs_[i].text + "###tab" + std::to_string(tabs_[i].id);
    const ImGuiTabItemFlags flags = (select_current && i == current_index_) ? ImGuiTabItemFlags_SetSelected : 0;
    if (ImGui::BeginTabItem(label.c_str(), tabs_closable_ ? &open : nullptr, flags)) {
      // a programmatic selection takes effect on the next frame, ignore the old tab until then
      if (!select_current && i != current_index_) {
        current_index_ = i;
        currentChanged(i);
      }
      ImGui::EndTabItem();
    }
    tabs_[i].rect = ImRect(ImGui::GetItemRectMin(), ImGui::GetItemRectMax());
    if (!tabs_[i].tooltip.empty()) ImGui::SetItemTooltip("%s", tabs_[i].tooltip.c_str());
    tabContextMenu(i);
    if (!open) close_index = i;
  }
  if (tabs_closable_) style.TabCloseButtonMinWidthUnselected = close_button_min_width;
  scroll_buttons_ ? endScrollableTabBar() : ImGui::EndTabBar();
  ImGui::PopID();
  if (close_index >= 0) tabCloseRequested(close_index);
}
