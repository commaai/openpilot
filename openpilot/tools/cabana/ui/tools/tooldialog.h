#pragma once

#include <cstdio>
#include <string>

#include "imgui.h"
#include "tools/cabana/core/observable.h"
#include "tools/cabana/ui/util.h"

// non-modal dialogs drawn by MainWindow every frame until closed
class ToolDialog {
public:
  virtual ~ToolDialog() = default;
  virtual bool draw() = 0;  // false once the dialog was closed

  Connections connections_;  // dies with the dialog

protected:
  void setTitle(const std::string &name) {
    char buf[32];
    snprintf(buf, sizeof(buf), "###tooldialog%p", (void *)this);
    title_ = name + buf;
  }

  // draw() body: `if (begin(size)) { content } return end();`
  bool begin(const ImVec2 &size) {
    if (!open_) return false;
    ImGui::SetNextWindowSize(size, ImGuiCond_Appearing);
    setNextWindowFloatsOut();
    began_ = true;
    return visible_ = ImGui::Begin(title_.c_str(), &open_, ImGuiWindowFlags_NoSavedSettings);
  }

  bool end() {
    if (!began_) return false;
    // Escape closes the dialog like QDialog, but not while a popup is open above it
    if (visible_ && ImGui::IsWindowFocused(ImGuiFocusedFlags_RootAndChildWindows) &&
        ImGui::IsKeyPressed(ImGuiKey_Escape, false) &&
        !ImGui::IsPopupOpen(nullptr, ImGuiPopupFlags_AnyPopupId | ImGuiPopupFlags_AnyPopupLevel)) {
      open_ = false;
    }
    ImGui::End();
    began_ = false;
    return open_;
  }

  std::string title_;
  bool open_ = true;

private:
  bool began_ = false, visible_ = false;
};
