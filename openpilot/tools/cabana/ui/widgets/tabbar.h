#pragma once

#include <string>
#include <vector>

#include "imgui.h"
#include "imgui_internal.h"

#include "tools/cabana/core/observable.h"

// QTabBar: tabs are closable when setTabsClosable(true)
class TabBar {
public:
  TabBar() = default;
  int addTab(const std::string &text);
  int count() const { return (int)tabs_.size(); }
  void setTabText(int index, const std::string &text) { if (index >= 0 && index < count()) tabs_[index].text = text; }
  const std::string &tabText(int index) const { return tabs_[index].text; }
  void setTabToolTip(int index, const std::string &tip) { if (index >= 0 && index < count()) tabs_[index].tooltip = tip; }
  void setTabData(int index, int data) { if (index >= 0 && index < count()) tabs_[index].data = data; }
  int tabData(int index) const { return index >= 0 && index < count() ? tabs_[index].data : 0; }
  int currentIndex() const { return current_index_; }
  void setCurrentIndex(int index);
  int tabAt(const ImVec2 &pos) const;  // -1 when no tab covers pos
  void removeTab(int index);
  void moveTab(int from, int to);
  void setAutoHide(bool hide) { auto_hide_ = hide; }
  void setTabsClosable(bool closable) { tabs_closable_ = closable; }  // off by default
  void setUsesScrollButtons(bool use) { scroll_buttons_ = use; }
  void draw();

  Observable<int> currentChanged;
  Observable<int> tabCloseRequested;
  Observable<int> tabContextMenu;  // emitted while drawing, right after the tab: open a context popup from it

private:
  struct Tab { std::string text; int data = 0; int id = 0; std::string tooltip; ImRect rect; };
  std::vector<Tab> tabs_;
  int current_index_ = -1;
  int next_id_ = 0;
  bool select_current_ = false;  // programmatic current change, applied at the next draw()
  bool auto_hide_ = false;
  bool tabs_closable_ = false;
  bool scroll_buttons_ = false;
};
