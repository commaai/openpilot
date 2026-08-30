#pragma once

#include <string>
#include <vector>

#include "imgui.h"
#include "imgui_internal.h"

// one line of the tip: [square] name <b>value</b> (min, max)
struct TipLine {
  bool has_marker = false;
  ImU32 marker = 0;
  std::string name;
  std::string bold;
  std::string rest;
};

class TipLabel {
public:
  void showText(const ImVec2 &pt, const std::vector<TipLine> &text, const ImRect &rect);
  void hide() { visible_ = false; }
  bool isVisible() const { return visible_; }
  void paintEvent();  // draws the tip on the foreground draw list; call once per frame

private:
  ImVec2 sizeHint() const;
  std::vector<TipLine> text_;
  ImVec2 pos_;
  ImVec2 size_;
  int margin_ = 2;  // 1 + PM_ToolTipLabelFrameWidth
  bool visible_ = false;
};
