#include "tools/cabana/ui/util.h"

#include <algorithm>
#include <cctype>
#include <cfloat>
#include <string>
#include <vector>

#include "imgui.h"
#include "imgui_internal.h"
#ifdef __APPLE__
#include <CoreFoundation/CoreFoundation.h>
#include <GLFW/glfw3.h>
extern "C" {
struct objc_object;
struct objc_selector;
objc_object *glfwGetCocoaWindow(GLFWwindow *window);
objc_selector *sel_registerName(const char *name);
void objc_msgSend(void);
}
#endif

#include "tools/cabana/ui/icons.h"

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

bool inputTextMultiline(const char *label, std::string *s, const ImVec2 &size, ImGuiInputTextFlags flags) {
  InputContext ctx{s, nullptr};
  return ImGui::InputTextMultiline(label, s->data(), s->capacity() + 1, size, flags | ImGuiInputTextFlags_CallbackResize,
                                   inputCallback, &ctx);
}

bool clearableInput(const char *label, std::string *s, const char *hint, ImGuiInputTextCallback validator) {
  bool changed = validatedInput(label, s, validator, hint);
  if (!s->empty()) {
    ImGui::SameLine(0.0f, 0.0f);
    ImGui::PushID(label);
    if (toolButton("clear", icon::X)) {
      s->clear();
      changed = true;
    }
    ImGui::PopID();
  }
  return changed;
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

void disabledItemTooltip(const char *text) {
  if (ImGui::IsItemHovered(ImGuiHoveredFlags_ForTooltip | ImGuiHoveredFlags_AllowWhenDisabled)) ImGui::SetTooltip("%s", text);
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

ImGuiWindow *topPopupWindow() {
  ImGuiContext &g = *GImGui;
  return g.OpenPopupStack.Size > 0 ? g.OpenPopupStack.back().Window : nullptr;
}

bool dialogEscapePressed() {
  return ImGui::IsKeyPressed(ImGuiKey_Escape, false) && topPopupWindow() == ImGui::GetCurrentWindow();
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
  if (rejected && dialogEscapePressed()) {
    *rejected = true;
    pressed = true;
  }
  return pressed;
}

int tableHeadersRow() {
  int clicked = -1;
  ImGui::TableNextRow(ImGuiTableRowFlags_Headers);
  for (int c = 0, count = ImGui::TableGetColumnCount(); c < count; ++c) {
    if (!ImGui::TableSetColumnIndex(c)) continue;
    const char *name = ImGui::TableGetColumnName(c);
    if (!name) name = "";
    const float offset = (ImGui::GetContentRegionAvail().x - ImGui::CalcTextSize(name).x) * 0.5f;
    if (offset > 0) ImGui::SetCursorPosX(ImGui::GetCursorPosX() + offset);
    ImGui::PushID(c);
    ImGui::TableHeader(name);
    // same timing as TableHeader's own TableOpenContextMenu, so a menu opened by the caller is opened last
    // and replaces the (disabled) table context menu in the popup stack
    if (ImGui::IsItemHovered() && ImGui::IsMouseReleased(ImGuiMouseButton_Right)) clicked = c;
    ImGui::PopID();
  }
  return clicked;
}

bool viewSelectable(const char *label, bool selected, ImGuiSelectableFlags flags, const ImVec2 &size) {
  ImGui::PushStyleColor(ImGuiCol_HeaderHovered, selected ? ImGui::GetColorU32(ImGuiCol_Header) : IM_COL32(0, 0, 0, 0));
  ImGui::PushStyleColor(ImGuiCol_HeaderActive, ImGui::GetColorU32(ImGuiCol_Header));
  const bool clicked = ImGui::Selectable(label, selected, flags, size);
  ImGui::PopStyleColor(2);
  return clicked;
}

bool checkBox(const char *label, bool *v) {
  const float box = 16.0f;
  ImGuiWindow *window = ImGui::GetCurrentWindow();
  if (window->SkipItems) return false;
  const ImGuiStyle &style = ImGui::GetStyle();
  const ImGuiID id = window->GetID(label);
  const ImVec2 label_size = ImGui::CalcTextSize(label, nullptr, true);
  const float frame_h = ImGui::GetFrameHeight();
  const ImVec2 pos = window->DC.CursorPos;
  const ImRect total_bb(pos, ImVec2(pos.x + box + (label_size.x > 0.0f ? style.ItemInnerSpacing.x + label_size.x : 0.0f), pos.y + frame_h));
  ImGui::ItemSize(total_bb, style.FramePadding.y);
  if (!ImGui::ItemAdd(total_bb, id)) return false;
  bool hovered, held;
  const bool pressed = ImGui::ButtonBehavior(total_bb, id, &hovered, &held);
  if (pressed) {
    *v = !*v;
    ImGui::MarkItemEdited(id);
  }
  const float y = pos.y + IM_TRUNC((frame_h - box) * 0.5f);
  const ImRect check_bb(ImVec2(pos.x, y), ImVec2(pos.x + box, y + box));
  ImGui::RenderNavCursor(total_bb, id);
  const ImU32 bg = ImGui::GetColorU32((held && hovered) ? ImGuiCol_FrameBgActive : hovered ? ImGuiCol_FrameBgHovered : ImGuiCol_FrameBg);
  ImGui::RenderFrame(check_bb.Min, check_bb.Max, bg, true, style.FrameRounding);
  if (*v) {
    const float pad = ImMax(1.0f, IM_TRUNC(box / 6.0f));
    ImGui::RenderCheckMark(window->DrawList, ImVec2(check_bb.Min.x + pad, check_bb.Min.y + pad), ImGui::GetColorU32(ImGuiCol_CheckMark), box - pad * 2.0f);
  }
  if (label_size.x > 0.0f) ImGui::RenderText(ImVec2(check_bb.Max.x + style.ItemInnerSpacing.x, pos.y + style.FramePadding.y), label);
  return pressed;
}

void alignRight(float width) {
  ImGui::SameLine();
  ImGui::SetCursorPosX(ImGui::GetCursorPosX() + std::max(0.0f, ImGui::GetContentRegionAvail().x - width));
}

void drawText(ImDrawList *dl, const ImRect &rect, const char *text, ImU32 col, ImFont *font, float font_size, const ImVec2 &align) {
  if (font == nullptr) font = ImGui::GetFont();
  if (font_size <= 0.0f) font_size = ImGui::GetFontSize();
  const ImVec2 size = font->CalcTextSizeA(font_size, FLT_MAX, 0.0f, text);
  const ImVec2 pos(rect.Min.x + (rect.GetWidth() - size.x) * align.x, rect.Min.y + (rect.GetHeight() - size.y) * align.y);
  dl->AddText(font, font_size, pos, col, text);
}

void drawElidedText(ImDrawList *dl, const ImRect &rect, const std::string &text, ImU32 col, bool align_right) {
  const ImVec2 size = ImGui::CalcTextSize(text.c_str());
  const float y = rect.Min.y + std::max(0.0f, (rect.GetHeight() - size.y) * 0.5f);
  if (size.x <= rect.GetWidth()) {
    dl->AddText(ImVec2(align_right ? rect.Max.x - size.x : rect.Min.x, y), col, text.c_str());
  } else {
    ImGui::PushStyleColor(ImGuiCol_Text, col);
    ImGui::RenderTextEllipsis(dl, ImVec2(rect.Min.x, y), ImVec2(rect.Max.x, y + size.y), rect.Max.x, text.c_str(), nullptr, &size);
    ImGui::PopStyleColor();
  }
}

float markerSize() { return ImGui::GetTextLineHeight() - 4; }

void drawColorMarker(ImDrawList *dl, const ImVec2 &pos, ImU32 col) {
  const float size = markerSize();
  dl->AddRectFilled(ImVec2(pos.x, pos.y + 2), ImVec2(pos.x + size, pos.y + 2 + size), col);
}

#ifdef __APPLE__
void setMacAppName(const char *name) {
  auto info = (CFMutableDictionaryRef)CFBundleGetInfoDictionary(CFBundleGetMainBundle());
  if (info == nullptr) return;
  CFStringRef value = CFStringCreateWithCString(kCFAllocatorDefault, name, kCFStringEncodingUTF8);
  CFDictionarySetValue(info, CFSTR("CFBundleName"), value);
  CFRelease(value);
}

bool isNativeFullScreen(GLFWwindow *window) {
  constexpr unsigned long NS_WINDOW_STYLE_MASK_FULL_SCREEN = 1ul << 14;
  objc_object *ns_window = glfwGetCocoaWindow(window);
  if (ns_window == nullptr) return false;
  auto styleMask = (unsigned long (*)(objc_object *, objc_selector *))objc_msgSend;
  return (styleMask(ns_window, sel_registerName("styleMask")) & NS_WINDOW_STYLE_MASK_FULL_SCREEN) != 0;
}

void toggleNativeFullScreen(GLFWwindow *window) {
  auto toggle = (void (*)(objc_object *, objc_selector *, objc_object *))objc_msgSend;
  toggle(glfwGetCocoaWindow(window), sel_registerName("toggleFullScreen:"), nullptr);
}
#endif

void setNextWindowFloatsOut() {
  ImGuiWindowClass window_class;
  window_class.ViewportFlagsOverrideSet = ImGuiViewportFlags_NoAutoMerge;
  ImGui::SetNextWindowClass(&window_class);
}

void setNextDialogWindow(const ImVec2 &size) {
  if (size.x > 0.0f || size.y > 0.0f) ImGui::SetNextWindowSize(size, ImGuiCond_Appearing);
  ImGui::SetNextWindowPos(ImGui::GetMainViewport()->GetCenter(), ImGuiCond_Appearing, ImVec2(0.5f, 0.5f));
  setNextWindowFloatsOut();
}

bool beginDialog(const char *id, PopupOwner *owner, const ImVec2 &size, ImGuiWindowFlags flags) {
  if (!owner->begin(id)) return false;
  setNextDialogWindow(size);
  return ImGui::BeginPopupModal(id, nullptr, flags | ImGuiWindowFlags_NoSavedSettings);
}

// tool bar

void beginToolbar() {
  // the items sit next to each other, the buttons only carry the auto raise margin
  ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(TOOLBAR_ITEM_SPACING, ImGui::GetStyle().ItemSpacing.y));
  ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(TOOLBAR_BUTTON_PADDING, ImGui::GetStyle().FramePadding.y));
}

void endToolbar() { ImGui::PopStyleVar(2); }

float toolbarButtonWidth(const std::string &label) {
  return ImGui::CalcTextSize(label.c_str(), nullptr, true).x + ImGui::GetStyle().FramePadding.x * 2;
}

static float toolbarGroupWidth(const std::vector<ToolbarItem> &items, size_t begin, size_t end) {
  float w = 0;
  for (size_t i = begin; i < end; ++i) w += items[i].width + (i > begin ? ImGui::GetStyle().ItemSpacing.x : 0);
  return w;
}

float toolbarWidth(const std::vector<ToolbarItem> &items, size_t spacer_index) {
  spacer_index = std::min(spacer_index, items.size());
  float w = toolbarGroupWidth(items, 0, spacer_index) + toolbarGroupWidth(items, spacer_index, items.size());
  if (spacer_index > 0 && spacer_index < items.size()) w += ImGui::GetStyle().ItemSpacing.x;
  return w;
}

void drawToolbar(const std::vector<ToolbarItem> &items, size_t spacer_index) {
  const ImGuiStyle &style = ImGui::GetStyle();
  spacer_index = std::min(spacer_index, items.size());
  const float right_width = toolbarGroupWidth(items, spacer_index, items.size());
  const float start_x = ImGui::GetCursorPosX();
  const float avail = ImGui::GetContentRegionAvail().x;
  const float right_edge = start_x + avail;
  const float extension_width = toolbarButtonWidth(icon::RAQUO);

  // when everything fits the spacer takes the slack, otherwise the extension button is reserved at the
  // right edge and the items are packed from the left until the next one does not fit
  const bool fits = toolbarWidth(items, spacer_index) <= avail;
  size_t visible = items.size();
  if (!fits) {
    const float usable = avail - (extension_width + style.ItemSpacing.x);
    float used = 0;
    for (visible = 0; visible < items.size(); ++visible) {
      const float w = items[visible].width + (visible ? style.ItemSpacing.x : 0);
      if (used + w > usable) break;
      used += w;
    }
  }

  for (size_t i = 0; i < visible; ++i) {
    if (i == 0) ImGui::SetCursorPosX(start_x);
    else if (fits && i == spacer_index) ImGui::SameLine(right_edge - right_width);
    else ImGui::SameLine();
    items[i].draw();
  }

  if (visible < items.size()) {
    // the extension button sits fully inside the toolbar: its right edge is the content region right edge
    const float extension_x = std::max(start_x, right_edge - extension_width);
    visible == 0 ? ImGui::SetCursorPosX(extension_x) : ImGui::SameLine(extension_x);
    if (ImGui::Button((std::string(icon::RAQUO) + "###toolbar_extension").c_str(), ImVec2(extension_width, 0)))
      ImGui::OpenPopup("toolbar_extension_menu");
    ImGui::SetItemTooltip("More");
    // the popup opens inward: its right edge is aligned with the button so it stays inside the window
    ImGui::SetNextWindowPos(ImVec2(ImGui::GetItemRectMax().x, ImGui::GetItemRectMax().y), ImGuiCond_Always, ImVec2(1, 0));
    if (ImGui::BeginPopup("toolbar_extension_menu")) {
      for (size_t i = visible; i < items.size(); ++i) {
        if (!items[i].in_menu) continue;
        if (items[i].menu_label.empty()) {
          items[i].draw();
        } else if (ImGui::MenuItem(items[i].menu_label.c_str(), nullptr, false, items[i].enabled)) {
          items[i].trigger();
        }
      }
      ImGui::EndPopup();
    }
  }
}

const float MENU_ARROW_SIZE = 6.0f;     // dropdown arrow on a menu button
const float MENU_ARROW_SPACING = 5.0f;  // gap between the label and the dropdown arrow

float menuButtonWidth(const std::string &text, bool bold) {
  if (bold) pushBoldFont();
  const float w = ImGui::CalcTextSize(text.c_str(), nullptr, true).x + MENU_ARROW_SPACING + MENU_ARROW_SIZE +
                  ImGui::GetStyle().FramePadding.x * 2;
  if (bold) popBoldFont();
  return w;
}

bool menuButton(const char *id, const std::string &text, const char *popup_id, bool bold, float width) {
  const ImGuiStyle &style = ImGui::GetStyle();
  const bool popup_open = ImGui::IsPopupOpen(popup_id);
  if (width <= 0.0f) width = menuButtonWidth(text, bold);
  // no frame, transparent until hovered; the button is drawn pressed while the menu is open. The menu opens
  // on press; a press while it is open toggles it closed (imgui closes the popup at the end of the frame of
  // a click outside it, so only open when it is not already open)
  if (bold) pushBoldFont();
  ImGui::PushStyleColor(ImGuiCol_Button, popup_open ? style.Colors[ImGuiCol_ButtonActive] : ImVec4(0, 0, 0, 0));
  ImGui::PushStyleVar(ImGuiStyleVar_FrameBorderSize, 0.0f);
  ImGui::PushStyleVar(ImGuiStyleVar_ButtonTextAlign, ImVec2(0.0f, 0.5f));
  const bool clicked = ImGui::ButtonEx((text + "###" + id).c_str(), ImVec2(width, 0.0f), ImGuiButtonFlags_PressedOnClick);
  ImGui::PopStyleVar(2);
  ImGui::PopStyleColor();
  const float text_width = ImGui::CalcTextSize(text.c_str(), nullptr, true).x;
  const float ascent = ImGui::GetFontBaked()->Ascent;
  if (bold) popBoldFont();
  // a 6 px arrow right after the text, sitting on the text baseline
  const ImVec2 min = ImGui::GetItemRectMin();
  const float x = min.x + style.FramePadding.x + text_width + MENU_ARROW_SPACING;
  const float baseline = min.y + style.FramePadding.y + ascent;
  ImGui::GetWindowDrawList()->AddTriangleFilled(ImVec2(x, baseline - MENU_ARROW_SIZE * 0.5f),
                                                ImVec2(x + MENU_ARROW_SIZE, baseline - MENU_ARROW_SIZE * 0.5f),
                                                ImVec2(x + MENU_ARROW_SIZE * 0.5f, baseline),
                                                ImGui::GetColorU32(ImGuiCol_TextDisabled));
  if (clicked && !popup_open) ImGui::OpenPopup(popup_id);
  // the menu drops down from below the button, not at the mouse cursor
  ImGui::SetNextWindowPos(ImVec2(min.x, ImGui::GetItemRectMax().y), ImGuiCond_Always);
  return clicked;
}
