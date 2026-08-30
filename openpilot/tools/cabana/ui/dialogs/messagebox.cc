#include "tools/cabana/ui/dialogs/messagebox.h"

#include <deque>

#include "imgui.h"
#include "imgui_internal.h"
#include "tools/cabana/ui/imgui_util.h"

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
// draw() is called both nested in a modal dialog and at the root level; only the level that opened the
// popup may submit it (opening at level 0 would make imgui close the parent modal).
ImGuiID g_popup_id = 0;
ImGuiID g_owner_id = 0;

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

bool isOpen() { return !g_queue.empty(); }

void draw() {
  if (g_queue.empty()) return;
  Box &box = g_queue.front();
  const std::string popup_id = box.title + "###MessageBox";
  ImGuiWindow *window = ImGui::GetCurrentWindowRead();  // GetCurrentWindow() would mark the fallback window as used
  if (g_popup_id == 0) {
    // a pending box may only be opened from the call nested in the top-most modal, or from any call when
    // there is no modal at all
    ImGuiWindow *modal = ImGui::GetTopMostPopupModal();
    if (modal != nullptr && modal != window) return;
    ImGui::OpenPopup(popup_id.c_str());
    g_popup_id = window->GetID(popup_id.c_str());
    g_owner_id = window->ID;
    g_show_details = false;
  } else if (g_owner_id != window->ID) {
    return;
  } else if (!ImGui::IsPopupOpen(g_popup_id, ImGuiPopupFlags_AnyPopupLevel)) {
    // reopen if imgui closed the popup underneath us (host window change)
    ImGui::OpenPopup(popup_id.c_str());
  }
  bool result = false, done = false;
  ImGui::SetNextWindowPos(ImGui::GetMainViewport()->GetCenter(), ImGuiCond_Appearing, ImVec2(0.5f, 0.5f));
  // AlwaysAutoResize sizes the popup from its contents; keep the title bar text from being clipped
  const ImGuiStyle &style = ImGui::GetStyle();
  const float min_width = ImGui::CalcTextSize(box.title.c_str()).x + style.FramePadding.x * 2 + style.WindowPadding.x * 2;
  ImGui::SetNextWindowSizeConstraints(ImVec2(min_width, 0.0f), ImVec2(FLT_MAX, FLT_MAX));
  setNextWindowFloatsOut();
  if (ImGui::BeginPopupModal(popup_id.c_str(), nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
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
      if (ImGui::Button(g_show_details ? "Hide Details..." : "Show Details...")) g_show_details = !g_show_details;
      ImGui::SameLine();
    }
    dialogButtons("OK", &result, &done, true, box.has_cancel ? "Cancel" : nullptr);
    if (ImGui::IsKeyPressed(ImGuiKey_Enter, false)) result = true;
    if (result) done = true;
    if (dialogEscapePressed()) done = true;
    if (done) ImGui::CloseCurrentPopup();
    ImGui::EndPopup();
  }
  if (done) {
    g_popup_id = 0;
    g_owner_id = 0;
    Box finished = std::move(g_queue.front());
    g_queue.pop_front();
    if (finished.on_result) finished.on_result(result);
  }
}

}  // namespace MessageBox
