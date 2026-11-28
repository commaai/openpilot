#pragma once

#include <string>
#include <utility>
#include <vector>

#include "imgui_internal.h"

// Dims the window and shows each widget's whatsThis text at its center; any click closes it.
// The texts are rich text: <b>, <br />, <span style="color:..;background-color:..">, &entities; and #rrggbb
// tokens (the video legend) are rendered, everything else is ignored.
class HelpOverlay {
public:
  void toggle();
  bool visible() const { return visible_; }
  // collected while the widgets draw, consumed by draw()
  void add(const std::string &text, const ImRect &rect);
  void draw();

private:
  struct Entry {
    std::string text;
    ImRect rect;
    ImGuiViewport *viewport;
  };
  std::vector<Entry> texts_;
  bool visible_ = false;
  int opened_frame_ = -1;
};
