#include "tools/cabana/ui/dialogs/messagebox.h"

#include <deque>

#include "imgui.h"
#include "imgui_internal.h"
#include "tools/cabana/ui/util.h"

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
PopupOwner g_owner;

void push(Box box) { g_queue.push_back(std::move(box)); }

std::function<void(bool)> wrap(std::function<void()> on_close) {
  if (!on_close) return nullptr;
  return [on_close = std::move(on_close)](bool) { on_close(); };
}

}  // namespace

void information(const std::string &title, const std::string &text, std::function<void()> on_close) {
  push({.title = title, .text = text, .on_result = wrap(std::move(on_close))});
}

void warning(const std::string &title, const std::string &text, const std::string &detailed_text,
             std::function<void()> on_close) {
  push({.title = title, .text = text, .detailed_text = detailed_text, .on_result = wrap(std::move(on_close))});
}

void question(const std::string &title, const std::string &text, std::function<void(bool)> on_result) {
  push({.title = title, .text = text, .has_cancel = true, .on_result = std::move(on_result)});
}

void draw() {
  if (g_queue.empty()) return;
  Box &box = g_queue.front();
  const std::string popup_id = box.title + "###MessageBox";
  const bool first = g_owner.popup_id == 0;
  if (!g_owner.begin(popup_id.c_str())) return;
  // AlwaysAutoResize sizes the popup from its contents; keep the title bar text from being clipped
  const ImGuiStyle &style = ImGui::GetStyle();
  const float min_width = ImGui::CalcTextSize(box.title.c_str()).x + style.FramePadding.x * 2 + style.WindowPadding.x * 2;
  ImGui::SetNextWindowSizeConstraints(ImVec2(min_width, 0.0f), ImVec2(FLT_MAX, FLT_MAX));
  setNextDialogWindow(ImVec2(0.0f, 0.0f));
  if (!ImGui::BeginPopupModal(popup_id.c_str(), nullptr, ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoSavedSettings)) return;
  if (first) g_show_details = false;
  bool result = false, done = false;
  ImGui::PushTextWrapPos(ImGui::GetCursorPos().x + 480.0f);
  ImGui::TextUnformatted(box.text.c_str());
  ImGui::PopTextWrapPos();
  if (g_show_details) {
    ImGui::InputTextMultiline("##details", box.detailed_text.data(), box.detailed_text.size() + 1,
                              ImVec2(480.0f, 160.0f), ImGuiInputTextFlags_ReadOnly);
  }
  ImGui::Separator();
  if (!box.detailed_text.empty()) {
    // the details button sits at the left of the button box
    if (ImGui::Button(g_show_details ? "Hide Details" : "Show Details")) g_show_details = !g_show_details;
    ImGui::SameLine();
  }
  dialogButtons("OK", &result, &done, true, box.has_cancel ? "Cancel" : nullptr);
  if (ImGui::IsKeyPressed(ImGuiKey_Enter, false) || ImGui::IsKeyPressed(ImGuiKey_KeypadEnter, false)) result = true;
  if (result) done = true;
  if (done) ImGui::CloseCurrentPopup();
  ImGui::EndPopup();
  if (done) {
    g_owner.reset();
    Box finished = std::move(g_queue.front());
    g_queue.pop_front();
    if (finished.on_result) finished.on_result(result);
  }
}

}  // namespace MessageBox
