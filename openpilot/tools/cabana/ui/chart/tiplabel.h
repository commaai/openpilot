#pragma once

#include <string>
#include <vector>

#include "imgui.h"
#include "imgui_internal.h"

// one line of the tip: Qt built these as rich text ("<span color>square</span>name<b>value</b>(min, max)")
struct TipLine {
  bool has_marker = false;
  ImU32 marker = 0;
  std::string name;
  std::string bold;
  std::string rest;
};

class TipLabel {
public:
  TipLabel();
  void showText(const ImVec2 &pt, const std::vector<TipLine> &text, const ImRect &rect);
  void hide() { visible_ = false; }
  bool isVisible() const { return visible_; }
  void paintEvent();  // draws the tip on the foreground draw list; call once per frame

private:
  ImVec2 sizeHint() const;
  std::vector<TipLine> text_;
  ImVec2 pos_;
  ImVec2 size_;
  int margin_ = 0;
  bool visible_ = false;
};
