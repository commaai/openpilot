#include "tools/cabana/ui/chart/signalselector.h"

#include <algorithm>
#include <cctype>
#include <cfloat>

#include "imgui.h"
#include "tools/cabana/streams/abstractstream.h"
#include "tools/cabana/ui/chart/chart.h"
#include "tools/cabana/ui/imgui_util.h"

SignalSelector::SignalSelector(std::string title) : title_(std::move(title)) {
  for (const auto &[id, _] : can->lastMessages()) {
    if (auto m = dbc()->msg(id)) {
      msgs_combo.push_back({m->name + " (" + id.toString() + ")", id});
    }
  }
  std::sort(msgs_combo.begin(), msgs_combo.end(), [](auto &a, auto &b) { return a.text < b.text; });
  msgs_combo_index_ = -1;
}

bool SignalSelector::draw() {
  if (!open_) return false;
  const std::string popup_id = title_ + "###SignalSelector";
  if (!show_) {
    ImGui::OpenPopup(popup_id.c_str());
    show_ = true;
  }
  ImGui::SetNextWindowSize(ImVec2(700.0f, 450.0f), ImGuiCond_Appearing);
  ImGui::SetNextWindowPos(ImGui::GetMainViewport()->GetCenter(), ImGuiCond_Appearing, ImVec2(0.5f, 0.5f));
  if (!ImGui::BeginPopupModal(popup_id.c_str(), nullptr, ImGuiWindowFlags_NoSavedSettings)) {
    open_ = false;
    return false;
  }

  const float btn_w = ImGui::GetFrameHeight() + 8.0f;
  const float column_w = (ImGui::GetContentRegionAvail().x - btn_w - ImGui::GetStyle().ItemSpacing.x * 2) / 2;
  // the selected list spans the combo row too; both lists end above the Ok/Cancel row
  const float lists_h = ImGui::GetContentRegionAvail().y - ImGui::GetFrameHeightWithSpacing() * 3;
  bool done = false;

  // left column
  ImGui::BeginGroup();
  ImGui::TextUnformatted("Available Signals");
  // the editable QComboBox (NoInsert) with its completer became a combo popup with a filter box
  const char *preview = msgs_combo_index_ >= 0 ? msgs_combo[msgs_combo_index_].text.c_str() : "Select a msg...";
  ImGui::SetNextItemWidth(column_w);
  if (ImGui::BeginCombo("##msgs_combo", preview)) {
    if (ImGui::IsWindowAppearing()) {
      msgs_combo_filter_.clear();  // QComboBox reopens showing the full list
      ImGui::SetKeyboardFocusHere();
    }
    ImGui::SetNextItemWidth(-FLT_MIN);
    inputText("##msgs_filter", &msgs_combo_filter_, "Select a msg...");
    std::string filter = msgs_combo_filter_;
    std::transform(filter.begin(), filter.end(), filter.begin(), [](unsigned char c) { return std::tolower(c); });
    for (int i = 0; i < (int)msgs_combo.size(); ++i) {
      std::string text = msgs_combo[i].text;
      std::transform(text.begin(), text.end(), text.begin(), [](unsigned char c) { return std::tolower(c); });
      if (!filter.empty() && text.find(filter) == std::string::npos) continue;
      if (ImGui::Selectable(msgs_combo[i].text.c_str(), i == msgs_combo_index_)) {
        msgs_combo_index_ = i;
        updateAvailableList(i);
        ImGui::CloseCurrentPopup();
      }
    }
    ImGui::EndCombo();
  }
  bool add_dbl = false;
  drawList("##available_list", available_list, &available_row_, false, &add_dbl, ImVec2(column_w, lists_h));
  ImGui::EndGroup();

  // buttons
  ImGui::SameLine();
  ImGui::BeginGroup();
  ImGui::Dummy(ImVec2(btn_w, (lists_h + ImGui::GetFrameHeightWithSpacing() * 2) / 2 - ImGui::GetFrameHeight()));
  ImGui::BeginDisabled(available_row_ == -1);
  bool add_clicked = ImGui::Button(icon::CHEVRON_RIGHT, ImVec2(btn_w, 0));
  ImGui::EndDisabled();
  ImGui::BeginDisabled(selected_row_ == -1);
  bool remove_clicked = ImGui::Button(icon::CHEVRON_LEFT, ImVec2(btn_w, 0));
  ImGui::EndDisabled();
  ImGui::EndGroup();

  // right column
  ImGui::SameLine();
  ImGui::BeginGroup();
  ImGui::TextUnformatted("Selected Signals");
  bool remove_dbl = false;
  drawList("##selected_list", selected_list, &selected_row_, true, &remove_dbl, ImVec2(column_w, lists_h + ImGui::GetFrameHeightWithSpacing()));
  // QDialogButtonBox: [Cancel] [Ok], right aligned
  const float buttons_w = 80.0f * 2 + ImGui::GetStyle().ItemSpacing.x;
  ImGui::SetCursorPosX(ImGui::GetCursorPosX() + std::max(0.0f, column_w - buttons_w));
  if (ImGui::Button("Cancel", ImVec2(80.0f, 0.0f)) || ImGui::IsKeyPressed(ImGuiKey_Escape, false)) done = true;
  ImGui::SameLine();
  if (ImGui::Button("OK", ImVec2(80.0f, 0.0f))) {
    done = true;
    accepted_ = true;
  }
  ImGui::EndGroup();

  if ((add_dbl || add_clicked) && available_row_ >= 0 && available_row_ < (int)available_list.size()) {
    add(&available_list[available_row_]);
  } else if ((remove_dbl || remove_clicked) && selected_row_ >= 0 && selected_row_ < (int)selected_list.size()) {
    remove(&selected_list[selected_row_]);
  }

  if (done) {
    open_ = false;
    ImGui::CloseCurrentPopup();
  }
  ImGui::EndPopup();
  return open_;
}

void SignalSelector::drawList(const char *id, std::vector<ListItem> &list, int *current_row, bool show_msg_name, bool *double_clicked, const ImVec2 &size) {
  if (!ImGui::BeginListBox(id, size)) return;
  const float marker = ImGui::GetTextLineHeight() - 4;
  for (int i = 0; i < (int)list.size(); ++i) {
    const auto &item = list[i];
    ImGui::PushID(i);
    const ImVec2 pos = ImGui::GetCursorScreenPos();
    if (ImGui::Selectable("##item", i == *current_row)) *current_row = i;
    if (ImGui::IsItemHovered() && ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left)) {
      *current_row = i;
      *double_clicked = true;
    }
    // label: colored square, signal name, then the message name/id in gray
    ImDrawList *dl = ImGui::GetWindowDrawList();
    float x = pos.x + 5;
    const auto &c = item.sig->color;
    dl->AddRectFilled(ImVec2(x, pos.y + 2), ImVec2(x + marker, pos.y + 2 + marker), IM_COL32(c.r, c.g, c.b, c.a));
    x += marker + 4;
    dl->AddText(ImVec2(x, pos.y), ImGui::GetColorU32(ImGuiCol_Text), item.sig->name.c_str());
    if (show_msg_name) {
      x += ImGui::CalcTextSize(item.sig->name.c_str()).x;
      std::string msg = " " + msgName(item.msg_id) + " " + item.msg_id.toString();
      dl->AddText(ImVec2(x, pos.y), ImGui::GetColorU32(ImGuiCol_TextDisabled), msg.c_str());
    }
    ImGui::PopID();
  }
  ImGui::EndListBox();
}

void SignalSelector::add(ListItem *item) {
  auto it = item;
  addItemToList(selected_list, it->msg_id, it->sig, true);
  available_list.erase(available_list.begin() + (it - available_list.data()));
  available_row_ = -1;
}

void SignalSelector::remove(ListItem *item) {
  auto it = item;
  if (msgs_combo_index_ >= 0 && it->msg_id == msgs_combo[msgs_combo_index_].id) {
    addItemToList(available_list, it->msg_id, it->sig);
  }
  selected_list.erase(selected_list.begin() + (it - selected_list.data()));
  selected_row_ = -1;
}

void SignalSelector::updateAvailableList(int index) {
  if (index == -1) return;
  available_list.clear();
  available_row_ = -1;
  MessageId msg_id = msgs_combo[index].id;
  auto selected_items = seletedItems();
  for (auto s : dbc()->msg(msg_id)->getSignals()) {
    bool is_selected = std::any_of(selected_items.begin(), selected_items.end(),
                                   [sig = s, &msg_id](auto it) { return it->msg_id == msg_id && it->sig == sig; });
    if (!is_selected) {
      addItemToList(available_list, msg_id, s);
    }
  }
}

void SignalSelector::addItemToList(std::vector<ListItem> &parent, const MessageId id, const cabana::Signal *sig, bool show_msg_name) {
  // the label (color square, name, gray msg name) is drawn in drawList; show_msg_name is implied by the list
  parent.emplace_back(id, sig);
}

std::vector<SignalSelector::ListItem *> SignalSelector::seletedItems() {
  std::vector<SignalSelector::ListItem *> ret;
  for (int i = 0; i < selected_list.size(); ++i) ret.push_back(&selected_list[i]);
  return ret;
}
