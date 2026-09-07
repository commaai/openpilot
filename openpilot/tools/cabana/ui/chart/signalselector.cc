#include "tools/cabana/ui/chart/signalselector.h"

#include <algorithm>
#include <cfloat>

#include "imgui.h"
#include "tools/cabana/streams/abstractstream.h"
#include "tools/cabana/ui/chart/chart.h"
#include "tools/cabana/ui/icons.h"
#include "tools/cabana/ui/util.h"
#include "tools/cabana/utils/strings.h"

SignalSelector::SignalSelector(std::string title) : title_(std::move(title)) {
  for (const auto &[id, _] : can->lastMessages()) {
    if (auto m = dbc()->msg(id)) {
      msgs_combo_.push_back({m->name + " (" + id.toString() + ")", id});
    }
  }
  std::sort(msgs_combo_.begin(), msgs_combo_.end(), [](auto &a, auto &b) { return a.text < b.text; });
}

bool SignalSelector::draw() {
  if (!open_) return false;
  const std::string popup_id = title_ + "###SignalSelector";
  if (!show_) {
    ImGui::OpenPopup(popup_id.c_str());
    show_ = true;
  }
  setNextDialogWindow(ImVec2(700.0f, 450.0f));
  if (!ImGui::BeginPopupModal(popup_id.c_str(), nullptr, ImGuiWindowFlags_NoSavedSettings)) {
    open_ = false;
    return false;
  }

  const float btn_w = iconButtonWidth();
  const float column_w = (ImGui::GetContentRegionAvail().x - btn_w - ImGui::GetStyle().ItemSpacing.x * 2) / 2;
  // the selected list spans the combo row too; both lists end above the Ok/Cancel row
  const float lists_h = ImGui::GetContentRegionAvail().y - ImGui::GetFrameHeightWithSpacing() * 3;

  ImGui::BeginGroup();
  ImGui::TextUnformatted("Available Signals");
  // a combo popup with a filter box
  const char *preview = msgs_combo_index_ >= 0 ? msgs_combo_[msgs_combo_index_].text.c_str() : "Select a msg...";
  ImGui::SetNextItemWidth(column_w);
  if (ImGui::BeginCombo("##msgs_combo", preview)) {
    if (ImGui::IsWindowAppearing()) {
      msgs_combo_filter_.clear();  // reopen showing the full list
      ImGui::SetKeyboardFocusHere();
    }
    ImGui::SetNextItemWidth(-FLT_MIN);
    inputText("##msgs_filter", &msgs_combo_filter_, "Select a msg...");
    for (int i = 0; i < (int)msgs_combo_.size(); ++i) {
      if (!msgs_combo_filter_.empty() && !utils::containsCI(msgs_combo_[i].text, msgs_combo_filter_)) continue;
      if (ImGui::Selectable(msgs_combo_[i].text.c_str(), i == msgs_combo_index_)) {
        msgs_combo_index_ = i;
        updateAvailableList(i);
        ImGui::CloseCurrentPopup();
      }
    }
    ImGui::EndCombo();
  }
  bool add_dbl = false;
  drawList("##available_list", available_list_, &available_row_, false, &add_dbl, ImVec2(column_w, lists_h));
  ImGui::EndGroup();

  ImGui::SameLine();
  ImGui::BeginGroup();
  ImGui::Dummy(ImVec2(btn_w, (lists_h + ImGui::GetFrameHeightWithSpacing() * 2) / 2 - ImGui::GetFrameHeight()));
  ImGui::BeginDisabled(available_row_ == -1);
  bool add_clicked = iconButton("add", icon::CHEVRON_RIGHT, "Add");
  ImGui::EndDisabled();
  ImGui::BeginDisabled(selected_row_ == -1);
  bool remove_clicked = iconButton("remove", icon::CHEVRON_LEFT, "Remove");
  ImGui::EndDisabled();
  ImGui::EndGroup();

  ImGui::SameLine();
  ImGui::BeginGroup();
  ImGui::TextUnformatted("Selected Signals");
  bool remove_dbl = false;
  drawList("##selected_list", selected_list_, &selected_row_, true, &remove_dbl, ImVec2(column_w, lists_h + ImGui::GetFrameHeightWithSpacing()));
  bool rejected = false;
  dialogButtons("OK", &accepted_, &rejected);
  const bool done = accepted_ || rejected;
  ImGui::EndGroup();

  if ((add_dbl || add_clicked) && available_row_ >= 0 && available_row_ < (int)available_list_.size()) {
    add(available_row_);
  } else if ((remove_dbl || remove_clicked) && selected_row_ >= 0 && selected_row_ < (int)selected_list_.size()) {
    remove(selected_row_);
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
    drawColorMarker(dl, ImVec2(x, pos.y), toImU32(item.sig->color));
    x += markerSize() + 4;
    dl->AddText(ImVec2(x, pos.y), ImGui::GetColorU32(ImGuiCol_Text), item.sig->name.c_str());
    if (show_msg_name) {
      x += ImGui::CalcTextSize(item.sig->name.c_str()).x;
      dl->AddText(ImVec2(x, pos.y), ImGui::GetColorU32(ImGuiCol_TextDisabled), msgLabel(item.msg_id).c_str());
    }
    ImGui::PopID();
  }
  ImGui::EndListBox();
}

void SignalSelector::add(int row) {
  const auto &item = available_list_[row];
  selected_list_.emplace_back(item.msg_id, item.sig);
  available_list_.erase(available_list_.begin() + row);
  available_row_ = -1;
}

void SignalSelector::remove(int row) {
  const auto &item = selected_list_[row];
  if (msgs_combo_index_ >= 0 && item.msg_id == msgs_combo_[msgs_combo_index_].id) {
    available_list_.emplace_back(item.msg_id, item.sig);
  }
  selected_list_.erase(selected_list_.begin() + row);
  selected_row_ = -1;
}

void SignalSelector::updateAvailableList(int index) {
  if (index == -1) return;
  available_list_.clear();
  available_row_ = -1;
  MessageId msg_id = msgs_combo_[index].id;
  for (auto s : dbc()->msg(msg_id)->getSignals()) {
    bool is_selected = std::any_of(selected_list_.begin(), selected_list_.end(),
                                   [sig = s, &msg_id](auto &it) { return it.msg_id == msg_id && it.sig == sig; });
    if (!is_selected) {
      available_list_.emplace_back(msg_id, s);
    }
  }
}
