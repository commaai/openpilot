#include "tools/cabana/ui/dialogs/messagebox.h"

#include <deque>

#include "imgui.h"
#include "imgui_internal.h"

namespace MessageBox {

namespace {

struct Box {
  std::string title;
  std::string text;
  std::string detailed_text;
  bool has_cancel = false;
  std::function<void(bool)> on_result;
};

std::deque<Box> g_queue;
bool g_show_details = false;

void push(Box box) { g_queue.push_back(std::move(box)); }

}  // namespace

void information(const std::string &title, const std::string &text) {
  push({.title = title, .text = text});
}

void warning(const std::string &title, const std::string &text, const std::string &detailed_text) {
  push({.title = title, .text = text, .detailed_text = detailed_text});
}

void question(const std::string &title, const std::string &text, std::function<void(bool)> on_result) {
  push({.title = title, .text = text, .has_cancel = true, .on_result = std::move(on_result)});
}

bool isOpen() { return !g_queue.empty(); }

void draw() {
  if (g_queue.empty()) return;
  Box &box = g_queue.front();
  const std::string popup_id = box.title + "###MessageBox";
  // reopen if imgui closed the popup underneath us (another level-0 popup, host window change)
  if (!ImGui::IsPopupOpen(popup_id.c_str())) {
    ImGui::OpenPopup(popup_id.c_str());
    g_show_details = false;
  }
  bool result = false, done = false;
  ImGui::SetNextWindowPos(ImGui::GetMainViewport()->GetCenter(), ImGuiCond_Appearing, ImVec2(0.5f, 0.5f));
  if (ImGui::BeginPopupModal(popup_id.c_str(), nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
    ImGui::PushTextWrapPos(ImGui::GetCursorPos().x + 480.0f);
    ImGui::TextUnformatted(box.text.c_str());
    ImGui::PopTextWrapPos();
    if (g_show_details) {
      ImGui::InputTextMultiline("##details", box.detailed_text.data(), box.detailed_text.size() + 1,
                                ImVec2(480.0f, 160.0f), ImGuiInputTextFlags_ReadOnly);
    }
    ImGui::Separator();
    if (ImGui::Button("OK", ImVec2(80.0f, 0.0f)) || ImGui::IsKeyPressed(ImGuiKey_Enter, false)) {
      result = true;
      done = true;
    }
    if (box.has_cancel) {
      ImGui::SameLine();
      if (ImGui::Button("Cancel", ImVec2(80.0f, 0.0f))) done = true;
    }
    if (ImGui::GetTopMostPopupModal() == ImGui::GetCurrentWindow() && ImGui::IsKeyPressed(ImGuiKey_Escape, false)) done = true;
    if (!box.detailed_text.empty()) {
      ImGui::SameLine();
      if (ImGui::Button(g_show_details ? "Hide Details..." : "Show Details...")) g_show_details = !g_show_details;
    }
    if (done) ImGui::CloseCurrentPopup();
    ImGui::EndPopup();
  }
  if (done) {
    Box finished = std::move(g_queue.front());
    g_queue.pop_front();
    if (finished.on_result) finished.on_result(result);
  }
}

}  // namespace MessageBox
