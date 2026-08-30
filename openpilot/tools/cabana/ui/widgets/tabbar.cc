#include "tools/cabana/ui/widgets/tabbar.h"

#include <algorithm>

int TabBar::addTab(const std::string &text) {
  tabs_.push_back({text, 0, next_id_++});
  int index = count() - 1;
  // the "x" close button is drawn by BeginTabItem(p_open) and reports through closeTabClicked()
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

void TabBar::draw() {
  if (auto_hide_ && count() < 2) return;  // auto hidden with fewer than two tabs
  if (!ImGui::BeginTabBar("##tabbar", scroll_buttons_ ? ImGuiTabBarFlags_FittingPolicyScroll : 0)) return;
  for (int i = 0; i < count(); ++i) {
    bool open = true;
    const std::string label = tabs_[i].text + "###tab" + std::to_string(tabs_[i].id);
    const ImGuiTabItemFlags flags = (select_current_ && i == current_index_) ? ImGuiTabItemFlags_SetSelected : 0;
    if (ImGui::BeginTabItem(label.c_str(), tabs_closable_ ? &open : nullptr, flags)) {
      if (i != current_index_) {
        current_index_ = i;
        currentChanged(i);
      }
      ImGui::EndTabItem();
    }
    tabs_[i].rect = ImRect(ImGui::GetItemRectMin(), ImGui::GetItemRectMax());
    if (!open) closeTabClicked(i);
  }
  select_current_ = false;
  ImGui::EndTabBar();
}
