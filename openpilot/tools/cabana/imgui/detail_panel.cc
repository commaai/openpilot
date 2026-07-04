#include "tools/cabana/imgui/app.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <optional>
#include <set>
#include <string>
#include <vector>

#include "tools/cabana/commands.h"
#include "tools/cabana/dbc/dbc.h"
#include "tools/cabana/dbc/dbcmanager.h"

// binary_view.cc / signal_view.cc -- owned by this workstream, not declared
// in app.h (see MIGRATION.md conventions).
void draw_binary_view(AppState &app);
std::set<const cabana::Signal *> binary_view_overlapping_signals(const MessageId &id);
void draw_signal_view(AppState &app);

namespace {

void draw_centered_text(const char *text) {
  const float width = ImGui::GetContentRegionAvail().x;
  ImGui::SetCursorPosX((width - ImGui::CalcTextSize(text).x) * 0.5f);
  ImGui::TextUnformatted(text);
}

void draw_shortcut_row(const char *title, const char *key) {
  const float center = ImGui::GetContentRegionAvail().x * 0.5f;
  const float title_w = ImGui::CalcTextSize(title).x;
  ImGui::SetCursorPosX(center - title_w - ImGui::GetStyle().ItemSpacing.x);
  ImGui::AlignTextToFramePadding();
  ImGui::TextDisabled("%s", title);
  ImGui::SameLine(center);
  ImGui::BeginDisabled();
  ImGui::Button(key);
  ImGui::EndDisabled();
}

// mirrors CenterWidget::createWelcomeWidget() in tools/cabana/detailwidget.cc
void draw_welcome() {
  const ImVec2 size = ImGui::GetContentRegionAvail();
  const float content_height = 50.0f + 4 * (ImGui::GetFrameHeight() + ImGui::GetStyle().ItemSpacing.y);
  ImGui::SetCursorPosY(std::max(0.0f, (size.y - content_height) * 0.5f));
  push_bold_font(50.0f);
  draw_centered_text("CABANA");
  pop_bold_font();
  ImGui::Spacing();
  ImGui::PushStyleColor(ImGuiCol_Text, ImGui::GetStyleColorVec4(ImGuiCol_TextDisabled));
  draw_centered_text("<- Select a message to view details");
  ImGui::PopStyleColor();
  ImGui::Spacing();
  draw_shortcut_row("Pause", "Space");
  draw_shortcut_row("Help", "F1");
  draw_shortcut_row("WhatsThis", "Shift+F1");
}

// Truncates with a trailing "..." if `text` doesn't fit in `max_width` --
// poor-man's ElidedLabel (tools/cabana/utils/elidedlabel.h) for the message
// name header.
std::string elide_text(const std::string &text, float max_width) {
  if (ImGui::CalcTextSize(text.c_str()).x <= max_width) return text;
  std::string s = text;
  while (!s.empty() && ImGui::CalcTextSize((s + "...").c_str()).x > max_width) s.pop_back();
  return s + "...";
}

// Hand-rolled tab button (Button + Tab/TabSelected theme colors) instead of
// ImGui::BeginTabBar()/BeginTabItem(): confirmed root cause (item 6 audit,
// timeboxed repro) is that ImGui's tab bar defers ImGuiTabItemFlags_SetSelected
// by one frame for a tab item that is *itself* brand new that same frame --
// the newly-created tab is appended to the tab bar's internal list only
// after that frame's selection resolution pass runs, so "which tab is
// current" (and therefore which one's content renders) lags one frame behind
// "which tab was just requested selected". Repro: a minimal floating window
// (temporarily added to app.cc, see report) with BeginTabBar/BeginTabItem
// that pushes one new tab per frame and always requests the newest one
// selected -- the pane drawn every frame is consistently the *previous*
// frame's newest tab, never the current one. Our real flow does exactly this
// (open_msg_tabs.push_back() + app.selected_msg_id both change on the same
// frame a not-yet-open message is selected), and headless `--output` only
// ever renders that one frame, so a native tab bar would capture the wrong
// (or textbook-uninitialized) tab as current every time. Buttons use the
// same Header/Selectable primitives the messages panel's row-highlight
// already renders correctly, and read app.selected_msg_id directly with no
// analogous "current tab" state of their own to desync.
bool draw_tab_button(const char *label, bool is_selected, bool *hovered_out = nullptr) {
  ImGui::PushStyleColor(ImGuiCol_Button, ImGui::GetStyleColorVec4(is_selected ? ImGuiCol_TabSelected : ImGuiCol_Tab));
  ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImGui::GetStyleColorVec4(ImGuiCol_TabHovered));
  ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImGui::GetStyleColorVec4(ImGuiCol_TabSelected));
  ImGui::PushStyleColor(ImGuiCol_Text, ImGui::GetStyleColorVec4(is_selected ? ImGuiCol_Text : ImGuiCol_TextDisabled));
  const bool clicked = ImGui::Button(label);
  ImGui::PopStyleColor(4);
  if (hovered_out != nullptr) *hovered_out = ImGui::IsItemHovered();
  return clicked;
}

// mirrors DetailWidget::findOrAddTab(): append the message if it isn't
// already an open tab. Qt scans back-to-front for an existing match; a
// forward scan is equivalent here since we never reorder tabs.
void ensure_tab_open(AppState &app, const MessageId &id) {
  if (std::find(app.open_msg_tabs.begin(), app.open_msg_tabs.end(), id) == app.open_msg_tabs.end()) {
    app.open_msg_tabs.push_back(id);
  }
}

// Tab strip of open messages -- mirrors DetailWidget's TabBar/findOrAddTab.
// Highlighting reads app.selected_msg_id directly every frame, so there's no
// separate "which tab is active" state to keep in sync: a change made
// elsewhere (e.g. a click in the messages panel) is reflected here for free,
// and clicking a tab button here just writes app.selected_msg_id back.
//
// Closing a tab selects its left neighbor (or the new first tab if it was
// leftmost), or clears the selection back to the welcome screen if it was
// the last tab open -- mirrors Qt's tabCloseRequested -> removeTab, whose
// automatic current-tab reassignment DetailWidget relies on via
// QTabBar::currentChanged.
void draw_tab_strip(AppState &app) {
  ensure_tab_open(app, *app.selected_msg_id);

  std::optional<size_t> close_index;
  ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(10.0f, 5.0f));
  for (size_t i = 0; i < app.open_msg_tabs.size(); ++i) {
    const MessageId id = app.open_msg_tabs[i];
    const bool is_selected = (id == *app.selected_msg_id);
    if (i > 0) ImGui::SameLine(0.0f, 2.0f);

    ImGui::PushID(static_cast<int>(i));
    // id.toString() ("source:hexaddr") matches Qt's tab text exactly.
    const std::string label = id.toString();
    bool hovered = false;
    if (draw_tab_button(label.c_str(), is_selected, &hovered)) {
      app.selected_msg_id = id;
    }
    if (hovered) ImGui::SetTooltip("%s", msgName(id).c_str());

    ImGui::SameLine(0.0f, 0.0f);
    ImGui::PushStyleColor(ImGuiCol_Button, ImGui::GetStyleColorVec4(is_selected ? ImGuiCol_TabSelected : ImGuiCol_Tab));
    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.85f, 0.35f, 0.3f, 1.0f));
    ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(0.75f, 0.25f, 0.2f, 1.0f));
    if (ImGui::SmallButton("x")) close_index = i;
    ImGui::PopStyleColor(3);
    ImGui::PopID();
  }
  ImGui::PopStyleVar();

  if (close_index.has_value()) {
    const size_t i = *close_index;
    const bool was_selected = (app.open_msg_tabs[i] == *app.selected_msg_id);
    app.open_msg_tabs.erase(app.open_msg_tabs.begin() + i);
    if (was_selected) {
      if (!app.open_msg_tabs.empty()) {
        const size_t new_idx = std::min(i > 0 ? i - 1 : 0, app.open_msg_tabs.size() - 1);
        app.selected_msg_id = app.open_msg_tabs[new_idx];
      } else {
        app.selected_msg_id.reset();
      }
    }
  }
}

// -- EditMessageDialog -------------------------------------------------------
// ImGui port of tools/cabana/detailwidget.{h,cc}'s EditMessageDialog, opened
// from DetailWidget::editMsg() (the toolbar's pencil-icon "Edit Message"
// action, added just before "Remove Message"). Also the path Qt uses to
// *define* a brand-new message for an id with no DBC entry yet: editMsg()
// doesn't check dbc()->msg(msg_id) != nullptr before opening, so a click on
// an undefined message's row opens the dialog pre-filled with
// msgName(msg_id) == UNTITLED ("untitled") and an empty node/comment; typing
// a real name and clicking OK pushes an EditMsgCommand that, per its ctor
// (commands.cc), records old_name empty and therefore has redo() define the
// message and undo() call dbc()->removeMsg(id) to undo the definition.

// NameValidator port: word chars only, space -> underscore (matches
// util.cc's `input.replace(' ', '_')` before the ^(\w+) regex validate).
// Duplicated from signal_view.cc's identical filter (anonymous-namespace,
// not shared across translation units) rather than exporting it -- same
// per-file duplication convention already used for the tab-button workaround
// (see stream_selector.cc's draw_selector_tab_button comment).
int edit_msg_name_char_filter(ImGuiInputTextCallbackData *data) {
  if (data->EventChar == ' ') {
    data->EventChar = '_';
    return 0;
  }
  if (data->EventChar < 256 && (std::isalnum(static_cast<unsigned char>(data->EventChar)) || data->EventChar == '_')) return 0;
  return 1;  // discard
}

bool equal_ignore_case(const std::string &a, const std::string &b) {
  if (a.size() != b.size()) return false;
  for (size_t i = 0; i < a.size(); ++i) {
    if (std::tolower(static_cast<unsigned char>(a[i])) != std::tolower(static_cast<unsigned char>(b[i]))) return false;
  }
  return true;
}

std::string trim(const std::string &s) {
  const size_t b = s.find_first_not_of(" \t\n\r");
  if (b == std::string::npos) return "";
  const size_t e = s.find_last_not_of(" \t\n\r");
  return s.substr(b, e - b + 1);
}

struct EditMsgDialogState {
  bool need_open = false;
  bool active = false;
  MessageId msg_id;
  std::string original_name;  // mirrors EditMessageDialog::original_name (title at open time)
  char name_buf[128] = {};
  char node_buf[128] = {};
  char comment_buf[1024] = {};  // fixed-size stand-in for QTextEdit's unbounded text; see report
  int size_val = 1;
  bool ok_enabled = true;
  std::string error;
};
EditMsgDialogState g_edit_msg;

// mirrors EditMessageDialog::validateName()
void validate_edit_msg_name() {
  const std::string name = g_edit_msg.name_buf;
  bool valid = !equal_ignore_case(name, UNTITLED);
  g_edit_msg.error.clear();
  if (!name.empty() && valid && name != g_edit_msg.original_name) {
    valid = dbc()->msg(g_edit_msg.msg_id.source, name) == nullptr;
    if (!valid) g_edit_msg.error = "Name already exists";
  }
  g_edit_msg.ok_enabled = valid;
}

// mirrors DetailWidget::editMsg() constructing EditMessageDialog: size
// defaults to the live message's byte count when there's no DBC definition
// yet (dlg's `size` ctor arg), node/comment are left blank in that case
// (Qt's `if (auto msg = dbc()->msg(msg_id))` guard around setting them).
void open_edit_message_dialog(const MessageId &id) {
  g_edit_msg = EditMsgDialogState{};
  g_edit_msg.msg_id = id;
  const cabana::Msg *msg = dbc()->msg(id);
  const int size = msg != nullptr ? static_cast<int>(msg->size) : static_cast<int>(can->lastMessage(id).dat.size());
  g_edit_msg.original_name = msgName(id);
  std::snprintf(g_edit_msg.name_buf, sizeof(g_edit_msg.name_buf), "%s", g_edit_msg.original_name.c_str());
  if (msg != nullptr) {
    std::snprintf(g_edit_msg.node_buf, sizeof(g_edit_msg.node_buf), "%s", msg->transmitter.c_str());
    std::snprintf(g_edit_msg.comment_buf, sizeof(g_edit_msg.comment_buf), "%s", msg->comment.c_str());
  }
  g_edit_msg.size_val = std::clamp(size, 1, CAN_MAX_DATA_BYTES);
  validate_edit_msg_name();
  g_edit_msg.need_open = true;
  g_edit_msg.active = true;
}

// mirrors EditMessageDialog's QFormLayout body + DetailWidget::editMsg()'s
// UndoStack::push(new EditMsgCommand(...)) on accept.
void draw_edit_message_dialog() {
  constexpr const char *kPopupId = "Edit Message##edit_msg_dialog";
  if (g_edit_msg.need_open) {
    ImGui::OpenPopup(kPopupId);
    g_edit_msg.need_open = false;
  }
  if (!g_edit_msg.active) return;

  ImGui::SetNextWindowPos(ImGui::GetMainViewport()->GetCenter(), ImGuiCond_Appearing, ImVec2(0.5f, 0.5f));
  ImGui::SetNextWindowSize(ImVec2(480.0f, 0.0f), ImGuiCond_Appearing);
  if (!ImGui::BeginPopupModal(kPopupId, nullptr, ImGuiWindowFlags_AlwaysAutoResize)) return;

  push_bold_font();
  ImGui::TextUnformatted(("Edit message: " + g_edit_msg.msg_id.toString()).c_str());
  pop_bold_font();
  ImGui::Separator();

  if (!g_edit_msg.error.empty()) {
    ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.85f, 0.2f, 0.2f, 1.0f));
    ImGui::TextUnformatted(g_edit_msg.error.c_str());
    ImGui::PopStyleColor();
  }

  constexpr float kLabelW = 90.0f;
  constexpr float kFieldW = 320.0f;

  ImGui::AlignTextToFramePadding();
  ImGui::TextUnformatted("Name");
  ImGui::SameLine(kLabelW);
  ImGui::SetNextItemWidth(kFieldW);
  if (ImGui::InputText("##edit_msg_name", g_edit_msg.name_buf, sizeof(g_edit_msg.name_buf),
                        ImGuiInputTextFlags_CallbackCharFilter, edit_msg_name_char_filter)) {
    validate_edit_msg_name();
  }

  ImGui::AlignTextToFramePadding();
  ImGui::TextUnformatted("Size");
  ImGui::SameLine(kLabelW);
  ImGui::SetNextItemWidth(kFieldW);
  ImGui::InputInt("##edit_msg_size", &g_edit_msg.size_val);
  g_edit_msg.size_val = std::clamp(g_edit_msg.size_val, 1, CAN_MAX_DATA_BYTES);

  ImGui::AlignTextToFramePadding();
  ImGui::TextUnformatted("Node");
  ImGui::SameLine(kLabelW);
  ImGui::SetNextItemWidth(kFieldW);
  ImGui::InputText("##edit_msg_node", g_edit_msg.node_buf, sizeof(g_edit_msg.node_buf),
                    ImGuiInputTextFlags_CallbackCharFilter, edit_msg_name_char_filter);

  ImGui::AlignTextToFramePadding();
  ImGui::TextUnformatted("Comment");
  ImGui::SameLine(kLabelW);
  ImGui::InputTextMultiline("##edit_msg_comment", g_edit_msg.comment_buf, sizeof(g_edit_msg.comment_buf), ImVec2(kFieldW, 80.0f));

  ImGui::Spacing();
  ImGui::Separator();
  ImGui::BeginDisabled(!g_edit_msg.ok_enabled);
  if (ImGui::Button("OK", ImVec2(100.0f, 0.0f))) {
    UndoStack::push(new EditMsgCommand(g_edit_msg.msg_id, trim(g_edit_msg.name_buf), static_cast<uint32_t>(g_edit_msg.size_val),
                                       trim(g_edit_msg.node_buf), trim(g_edit_msg.comment_buf)));
    g_edit_msg.active = false;
    ImGui::CloseCurrentPopup();
  }
  ImGui::EndDisabled();
  ImGui::SameLine();
  if (ImGui::Button("Cancel", ImVec2(100.0f, 0.0f)) || ImGui::IsKeyPressed(ImGuiKey_Escape, false)) {
    g_edit_msg.active = false;
    ImGui::CloseCurrentPopup();
  }

  ImGui::EndPopup();
}

// mirrors DetailWidget::refresh(): name/id/freq/count line plus warnings
// (undefined message, size mismatch, overlapping signal bits).
void draw_header(const MessageId &id) {
  const cabana::Msg *msg = dbc()->msg(id);

  std::vector<std::string> warnings;
  if (msg != nullptr) {
    if (id.source == INVALID_SOURCE) {
      warnings.push_back("No messages received.");
    } else if (msg->size != can->lastMessage(id).dat.size()) {
      char buf[128];
      snprintf(buf, sizeof(buf), "Message size (%u) is incorrect.", msg->size);
      warnings.emplace_back(buf);
    }
    for (const cabana::Signal *s : binary_view_overlapping_signals(id)) {
      warnings.push_back(s->name + " has overlapping bits.");
    }
  }

  // "Edit Message" / "Remove Message" -- mirrors DetailWidget::createToolBar()'s
  // pencil (editMsg() -> EditMessageDialog -> UndoStack::push(EditMsgCommand))
  // and x-lg (removeMsg() -> UndoStack::push(RemoveMsgCommand)) toolbar
  // actions, in the same order Qt adds them. Edit has no enabled/disabled
  // wiring in the Qt reference (it's how an undefined message gets defined),
  // Remove is disabled exactly like action_remove_msg when the message has
  // no DBC definition. Right-anchored on the name line since this port has
  // no separate per-message toolbar row.
  const float avail_w = ImGui::GetContentRegionAvail().x;
  const float edit_btn_w = ImGui::CalcTextSize("Edit Msg").x + ImGui::GetStyle().FramePadding.x * 2.0f + 8.0f;
  const float remove_btn_w = ImGui::CalcTextSize("Remove Msg").x + ImGui::GetStyle().FramePadding.x * 2.0f + 8.0f;
  const float buttons_w = edit_btn_w + remove_btn_w + ImGui::GetStyle().ItemSpacing.x;

  const std::string name_text = msg != nullptr ? (msg->name + " (" + msg->transmitter + ")") : msgName(id);
  push_bold_font();
  const std::string elided = elide_text(name_text, std::max(10.0f, avail_w - buttons_w - ImGui::GetStyle().ItemSpacing.x));
  ImGui::TextUnformatted(elided.c_str());
  pop_bold_font();
  if (ImGui::IsItemHovered()) ImGui::SetTooltip("%s", name_text.c_str());

  ImGui::SameLine(std::max(0.0f, avail_w - buttons_w));
  if (ImGui::SmallButton("Edit Msg")) {
    open_edit_message_dialog(id);
  }
  if (ImGui::IsItemHovered()) ImGui::SetTooltip("Edit Message");

  ImGui::SameLine(0.0f, ImGui::GetStyle().ItemSpacing.x);
  ImGui::BeginDisabled(msg == nullptr);
  if (ImGui::SmallButton("Remove Msg")) {
    UndoStack::push(new RemoveMsgCommand(id));
  }
  ImGui::EndDisabled();
  if (ImGui::IsItemHovered()) ImGui::SetTooltip("Remove Message");

  std::string freq_label = "N/A";
  std::string count_label = "N/A";
  if (id.source != INVALID_SOURCE) {
    const CanData &last = can->lastMessage(id);
    char buf[32];
    if (last.freq > 0) {
      if (last.freq >= 0.95) snprintf(buf, sizeof(buf), "%.0f Hz", std::nearbyint(last.freq));
      else snprintf(buf, sizeof(buf), "%.2f Hz", last.freq);
    } else {
      snprintf(buf, sizeof(buf), "-- Hz");
    }
    freq_label = buf;
    count_label = std::to_string(last.count);
  }
  ImGui::TextDisabled("%s  |  %s  |  %s msgs", id.toString().c_str(), freq_label.c_str(), count_label.c_str());

  if (!warnings.empty()) {
    ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.85f, 0.55f, 0.15f, 1.0f));
    for (const std::string &w : warnings) {
      ImGui::TextWrapped("! %s", w.c_str());
    }
    ImGui::PopStyleColor();
  }
}

// "Msg" inner-tab content: binary view on top (mirrors DetailWidget's
// splitter top pane), a thin separator, then the signal editor (SignalView)
// filling the rest of the pane (mirrors the splitter's bottom pane).
void draw_msg_tab(AppState &app) {
  const float avail_h = ImGui::GetContentRegionAvail().y;
  const float binary_h = std::clamp(avail_h * 0.5f, 140.0f, 420.0f);
  ImGui::BeginChild("##binary_view_area", ImVec2(0.0f, binary_h), ImGuiChildFlags_Borders);
  draw_binary_view(app);
  ImGui::EndChild();

  ImGui::Separator();
  draw_signal_view(app);
}

// mirrors DetailWidget's inner QTabWidget: tab_widget->addTab(splitter, ...,
// "&Msg") then addTab(history_log, ..., "&Logs"). Qt places these on the
// South edge (setTabPosition(South)) with a borderless pane; hand-rolled
// buttons (see draw_tab_button) render on top here instead -- both the
// bottom placement and the native TabBar are accepted deviations, see
// report. Selection is local static state (there's only ever one Detail
// panel), mirroring tab_widget->currentIndex().
void draw_inner_tabs(AppState &app) {
  static int active_inner_tab = 0;  // 0 = Msg, 1 = Logs

  ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(12.0f, 5.0f));
  if (draw_tab_button("Msg", active_inner_tab == 0)) active_inner_tab = 0;
  ImGui::SameLine(0.0f, 2.0f);
  if (draw_tab_button("Logs", active_inner_tab == 1)) active_inner_tab = 1;
  ImGui::PopStyleVar();
  ImGui::Separator();

  if (active_inner_tab == 0) {
    draw_msg_tab(app);
  } else {
    draw_history_log(app);
  }
}

}  // namespace

void draw_detail_panel(AppState &app) {
  ImGui::PushStyleColor(ImGuiCol_WindowBg, ImGui::GetStyleColorVec4(ImGuiCol_ChildBg));
  if (ImGui::Begin(CENTER_WINDOW_TITLE, nullptr, ImGuiWindowFlags_NoCollapse)) {
    if (!app.selected_msg_id) {
      draw_welcome();
    } else {
      draw_tab_strip(app);
      if (!app.selected_msg_id) {
        // Closing the last open tab clears the selection -- mirrors
        // CenterWidget::clear() falling back to the welcome widget.
        draw_welcome();
      } else {
        ImGui::Separator();
        draw_header(*app.selected_msg_id);
        ImGui::Separator();
        draw_inner_tabs(app);
      }
    }
  }
  ImGui::End();
  ImGui::PopStyleColor();

  // Modal: independent of the Detail window's Begin/End (BeginPopupModal
  // opens its own top-level ImGui window and blocks input to everything
  // else, mirroring the Qt QDialog::exec() the button above used to trigger
  // via editMsg()), so draw it unconditionally every frame like the other
  // app-level dialogs in app.cc's draw_ui().
  draw_edit_message_dialog();
}
