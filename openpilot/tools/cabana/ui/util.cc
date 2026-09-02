#include "tools/cabana/ui/util.h"

#include <cctype>
#include <string>
#include <vector>

#include "imgui.h"
#include "imgui_internal.h"
#ifdef __APPLE__
#include <CoreFoundation/CoreFoundation.h>
#endif

int inputCallback(ImGuiInputTextCallbackData *data) {
  auto *ctx = static_cast<InputContext *>(data->UserData);
  if (data->EventFlag == ImGuiInputTextFlags_CallbackCharFilter) {
    return ctx->validator ? ctx->validator(data) : 0;
  }
  if (data->EventFlag == ImGuiInputTextFlags_CallbackEdit) {
    if (ctx->validate(std::string(data->Buf, data->BufTextLen)) == ValidState::Invalid) {
      data->DeleteChars(0, data->BufTextLen);
      data->InsertChars(0, ctx->last_valid->c_str());
    }
    return 0;
  }
  if (data->EventFlag == ImGuiInputTextFlags_CallbackResize) {
    ctx->str->resize(data->BufTextLen);
    data->Buf = ctx->str->data();
  }
  return 0;
}

bool validatedInput(const char *label, std::string *s, ImGuiInputTextCallback validator, const char *hint,
                    ImGuiInputTextFlags flags) {
  InputContext ctx{s, validator};
  flags |= ImGuiInputTextFlags_CallbackResize;
  if (validator) flags |= ImGuiInputTextFlags_CallbackCharFilter;
  return ImGui::InputTextWithHint(label, hint, s->data(), s->capacity() + 1, flags, inputCallback, &ctx);
}

bool comboBox(const char *label, int *index, const std::vector<std::string> &items) {
  bool changed = false;
  const int count = (int)items.size();
  if (ImGui::BeginCombo(label, *index >= 0 && *index < count ? items[*index].c_str() : "")) {
    for (int i = 0; i < count; ++i) {
      ImGui::PushID(i);
      if (ImGui::Selectable(items[i].c_str(), i == *index) && *index != i) {
        *index = i;
        changed = true;
      }
      if (i == *index) ImGui::SetItemDefaultFocus();
      ImGui::PopID();
    }
    ImGui::EndCombo();
  }
  return changed;
}

bool validatedText(const char *label, std::string *s, ValidState (*validate)(const std::string &),
                   const char *hint, ImGuiInputTextCallback filter) {
  const std::string last_valid = *s;  // a refused edit never reaches *s
  InputContext ctx{s, filter, validate, &last_valid};
  ImGuiInputTextFlags flags = ImGuiInputTextFlags_CallbackResize | ImGuiInputTextFlags_CallbackEdit;
  if (filter) flags |= ImGuiInputTextFlags_CallbackCharFilter;
  ImGui::InputTextWithHint(label, hint, s->data(), s->capacity() + 1, flags, inputCallback, &ctx);
  return *s != last_valid;
}

int nameValidator(ImGuiInputTextCallbackData *data) {
  // [A-Za-z0-9_], spaces rewritten to '_'
  if (data->EventChar == ' ') {
    data->EventChar = '_';
    return 0;
  }
  return (data->EventChar < 128 && (std::isalnum((int)data->EventChar) || data->EventChar == '_')) ? 0 : 1;
}

int nodeValidator(ImGuiInputTextCallbackData *data) {
  // \w+(,\w+)*
  return (data->EventChar < 128 && (std::isalnum((int)data->EventChar) || data->EventChar == '_' || data->EventChar == ',')) ? 0 : 1;
}

int doubleValidator(ImGuiInputTextCallbackData *data) {
  // C-locale floating-point
  const ImWchar c = data->EventChar;
  return (c < 128 && (std::isdigit((int)c) || c == '+' || c == '-' || c == '.' || c == 'e' || c == 'E')) ? 0 : 1;
}

int ipValidator(ImGuiInputTextCallbackData *data) {
  // [0-9.]
  const ImWchar c = data->EventChar;
  return ((c >= '0' && c <= '9') || c == '.') ? 0 : 1;
}

int nonWhitespaceValidator(ImGuiInputTextCallbackData *data) {
  // \S+
  return (data->EventChar < 128 && std::isspace((int)data->EventChar)) ? 1 : 0;
}

bool toolButton(const char *id, const char *icon, const char *tooltip, const char *text) {
  std::string label = text && *text ? std::string(icon) + " " + text + "###" + id : std::string(icon) + "###" + id;
  // no frame, transparent until hovered
  ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0, 0, 0, 0));
  ImGui::PushStyleVar(ImGuiStyleVar_FrameBorderSize, 0.0f);
  bool clicked = ImGui::Button(label.c_str());
  ImGui::PopStyleVar();
  ImGui::PopStyleColor();
  if (tooltip && *tooltip) ImGui::SetItemTooltip("%s", tooltip);
  return clicked;
}

bool radioMenuItem(const char *label, bool checked, float width) {
  const float indent = ImGui::GetFontSize();
  const ImVec2 pos = ImGui::GetCursorScreenPos();
  const bool clicked = ImGui::Selectable((std::string("##") + label).c_str(), false, ImGuiSelectableFlags_None,
                                         ImVec2(ImMax(width, ImGui::GetContentRegionAvail().x), 0.0f));
  const ImU32 color = ImGui::GetColorU32(ImGuiCol_Text);
  ImDrawList *painter = ImGui::GetWindowDrawList();
  if (checked) ImGui::RenderBullet(painter, ImVec2(pos.x + indent / 2, pos.y + ImGui::GetTextLineHeight() / 2), color);
  painter->AddText(ImVec2(pos.x + indent, pos.y), color, label);
  return clicked;
}

bool PopupOwner::begin(const char *id) {
  ImGuiWindow *window = ImGui::GetCurrentWindowRead();  // GetCurrentWindow() would mark the fallback window as used
  if (popup_id == 0) {
    // a pending popup may only be opened from the call nested in the top-most modal, or from any call
    // when there is no modal at all
    ImGuiWindow *modal = ImGui::GetTopMostPopupModal();
    if (modal != nullptr && modal != window) return false;
    ImGui::OpenPopup(id);
    popup_id = window->GetID(id);
    owner_id = window->ID;
  } else if (owner_id != window->ID) {
    return false;
  } else if (!ImGui::IsPopupOpen(popup_id, ImGuiPopupFlags_AnyPopupLevel)) {
    // reopen if imgui closed the popup underneath us (host window change)
    ImGui::OpenPopup(id);
  }
  return true;
}

bool dialogEscapePressed() {
  if (!ImGui::IsKeyPressed(ImGuiKey_Escape, false)) return false;
  ImGuiContext &g = *GImGui;
  return g.OpenPopupStack.Size > 0 && g.OpenPopupStack.back().Window == ImGui::GetCurrentWindow();
}

bool dialogButtons(const char *accept_label, bool *accepted, bool *rejected, bool accept_enabled,
                   const char *reject_label) {
  const float button_width = 80.0f;
  const int count = reject_label ? 2 : 1;
  const float total = button_width * count + ImGui::GetStyle().ItemSpacing.x * (count - 1);
  const float avail = ImGui::GetContentRegionAvail().x;
  if (avail > total) ImGui::SetCursorPosX(ImGui::GetCursorPosX() + avail - total);
  bool pressed = false;
  if (reject_label) {
    if (ImGui::Button(reject_label, ImVec2(button_width, 0.0f))) {
      if (rejected) *rejected = true;
      pressed = true;
    }
    ImGui::SameLine();
  }
  ImGui::BeginDisabled(!accept_enabled);
  if (ImGui::Button(accept_label, ImVec2(button_width, 0.0f))) {
    if (accepted) *accepted = true;
    pressed = true;
  }
  ImGui::EndDisabled();
  return pressed;
}

void tableHeadersRow() {
  ImGui::TableNextRow(ImGuiTableRowFlags_Headers);
  for (int c = 0, count = ImGui::TableGetColumnCount(); c < count; ++c) {
    if (!ImGui::TableSetColumnIndex(c)) continue;
    const char *name = ImGui::TableGetColumnName(c);
    if (!name) name = "";
    const float offset = (ImGui::GetContentRegionAvail().x - ImGui::CalcTextSize(name).x) * 0.5f;
    if (offset > 0) ImGui::SetCursorPosX(ImGui::GetCursorPosX() + offset);
    ImGui::PushID(c);
    ImGui::TableHeader(name);
    ImGui::PopID();
  }
}

bool viewSelectable(const char *label, bool selected, ImGuiSelectableFlags flags, const ImVec2 &size) {
  ImGui::PushStyleColor(ImGuiCol_HeaderHovered, selected ? ImGui::GetColorU32(ImGuiCol_Header) : IM_COL32(0, 0, 0, 0));
  ImGui::PushStyleColor(ImGuiCol_HeaderActive, ImGui::GetColorU32(ImGuiCol_Header));
  const bool clicked = ImGui::Selectable(label, selected, flags, size);
  ImGui::PopStyleColor(2);
  return clicked;
}

bool checkBox(const char *label, bool *v) {
  ImGuiWindow *window = ImGui::GetCurrentWindow();
  if (window->SkipItems) return false;
  const ImGuiStyle &style = ImGui::GetStyle();
  const ImGuiID id = window->GetID(label);
  const ImVec2 label_size = ImGui::CalcTextSize(label, nullptr, true);
  const float frame_h = ImGui::GetFrameHeight();
  const ImVec2 pos = window->DC.CursorPos;
  const ImRect total_bb(pos, ImVec2(pos.x + CHECKBOX_SIZE + (label_size.x > 0.0f ? style.ItemInnerSpacing.x + label_size.x : 0.0f), pos.y + frame_h));
  ImGui::ItemSize(total_bb, style.FramePadding.y);
  if (!ImGui::ItemAdd(total_bb, id)) return false;
  bool hovered, held;
  const bool pressed = ImGui::ButtonBehavior(total_bb, id, &hovered, &held);
  if (pressed) {
    *v = !*v;
    ImGui::MarkItemEdited(id);
  }
  const float y = pos.y + IM_TRUNC((frame_h - CHECKBOX_SIZE) * 0.5f);
  const ImRect check_bb(ImVec2(pos.x, y), ImVec2(pos.x + CHECKBOX_SIZE, y + CHECKBOX_SIZE));
  ImGui::RenderNavCursor(total_bb, id);
  const ImU32 bg = ImGui::GetColorU32((held && hovered) ? ImGuiCol_FrameBgActive : hovered ? ImGuiCol_FrameBgHovered : ImGuiCol_FrameBg);
  ImGui::RenderFrame(check_bb.Min, check_bb.Max, bg, true, style.FrameRounding);
  if (*v) {
    const float pad = ImMax(1.0f, IM_TRUNC(CHECKBOX_SIZE / 6.0f));
    ImGui::RenderCheckMark(window->DrawList, ImVec2(check_bb.Min.x + pad, check_bb.Min.y + pad), ImGui::GetColorU32(ImGuiCol_CheckMark), CHECKBOX_SIZE - pad * 2.0f);
  }
  if (label_size.x > 0.0f) ImGui::RenderText(ImVec2(check_bb.Max.x + style.ItemInnerSpacing.x, pos.y + style.FramePadding.y), label);
  return pressed;
}

#ifdef __APPLE__
void setMacAppName(const char *name) {
  auto info = (CFMutableDictionaryRef)CFBundleGetInfoDictionary(CFBundleGetMainBundle());
  if (info == nullptr) return;
  CFStringRef value = CFStringCreateWithCString(kCFAllocatorDefault, name, kCFStringEncodingUTF8);
  CFDictionarySetValue(info, CFSTR("CFBundleName"), value);
  CFRelease(value);
}
#endif

void setNextWindowFloatsOut() {
  ImGuiWindowClass window_class;
  window_class.ViewportFlagsOverrideSet = ImGuiViewportFlags_NoAutoMerge;
  ImGui::SetNextWindowClass(&window_class);
}

bool beginDialog(const char *id, PopupOwner *owner, const ImVec2 &size) {
  if (!owner->begin(id)) return false;
  ImGui::SetNextWindowSize(size, ImGuiCond_Appearing);
  ImGui::SetNextWindowPos(ImGui::GetMainViewport()->GetCenter(), ImGuiCond_Appearing, ImVec2(0.5f, 0.5f));
  setNextWindowFloatsOut();
  return ImGui::BeginPopupModal(id, nullptr, ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoSavedSettings);
}
