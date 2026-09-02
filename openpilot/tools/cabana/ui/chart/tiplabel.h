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
  void draw();  // draws the tip on the foreground draw list; call once per frame

private:
  // lays the lines out from origin, drawing them when p is given; returns the size of the text block
  ImVec2 layoutLines(ImDrawList *p, const ImVec2 &origin, ImU32 fg) const;
  ImVec2 sizeHint() const;

  static constexpr float MARGIN = 2.0f;  // 1 + PM_ToolTipLabelFrameWidth
  std::vector<TipLine> text_;
  ImVec2 pos_;
  ImVec2 size_;
  bool visible_ = false;
};
