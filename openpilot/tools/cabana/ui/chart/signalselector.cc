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
  std::set<MessageId> ids;
  for (const auto &[id, _] : can->eventsMap()) ids.insert(id);
  for (const auto &[id, _] : can->lastMessages()) ids.insert(id);
  for (const auto &id : ids) {
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
  ImGui::SetNextItemWidth(column_w);
  if (ImGui::IsWindowAppearing()) ImGui::SetKeyboardFocusHere();
  if (inputText("##signal_search", &msgs_combo_filter_, "Search signals, messages, or IDs...")) available_dirty_ = true;
  if (available_dirty_) { updateAvailableList(); available_dirty_ = false; }
  bool add_dbl = false;
  drawList("##available_list", available_list_, &available_row_, true, &add_dbl, ImVec2(column_w, lists_h));
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
  ImGuiListClipper clipper;
  clipper.Begin(list.size());
  while (clipper.Step()) for (int i = clipper.DisplayStart; i < clipper.DisplayEnd; ++i) {
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
    drawColorMarker(dl, ImVec2(x, pos.y), toImU32(item.sig ? item.sig->color : CabanaColor{0, 114, 178}));
    x += markerSize() + 4;
    dl->AddText(ImVec2(x, pos.y), ImGui::GetColorU32(ImGuiCol_Text), item.name().c_str());
    if (show_msg_name && item.path.empty()) {
      x += ImGui::CalcTextSize(item.name().c_str()).x;
      dl->AddText(ImVec2(x, pos.y), ImGui::GetColorU32(ImGuiCol_TextDisabled), msgLabel(item.msg_id).c_str());
    }
    ImGui::PopID();
  }
  ImGui::EndListBox();
}

void SignalSelector::add(int row) {
  available_dirty_ = true;
  const auto &item = available_list_[row];
  selected_list_.push_back(item);
  available_list_.erase(available_list_.begin() + row);
  available_row_ = -1;
}

void SignalSelector::remove(int row) {
  available_dirty_ = true;
  selected_list_.erase(selected_list_.begin() + row);
  selected_row_ = -1;
}

void SignalSelector::updateAvailableList() {
  // Rebuild only when the query or selection changes; preserve the selected row otherwise.
  std::vector<ListItem> available;
  for (const auto &msg : msgs_combo_) {
    auto *message = dbc()->msg(msg.id);
    if (!message) continue;
    for (auto *sig : message->getSignals()) {
      if (!msgs_combo_filter_.empty() && !utils::containsCI(sig->name + " " + msg.text, msgs_combo_filter_)) continue;
      if (std::none_of(selected_list_.begin(), selected_list_.end(), [&](const auto &item) {
            return item.msg_id == msg.id && item.sig == sig;
          })) available.emplace_back(msg.id, sig);
    }
  }
  for (const auto &entry : can->telemetry) {
    const auto &path = entry.first;
    if (!utils::containsCI(path, msgs_combo_filter_)) continue;
    if (std::none_of(selected_list_.begin(), selected_list_.end(), [&](const auto &item) { return item.path == path; }))
      available.emplace_back(path);
  }
  if (available.size() != available_list_.size() || !std::equal(available.begin(), available.end(), available_list_.begin(),
      [](const auto &a, const auto &b) { return a.path == b.path && a.msg_id == b.msg_id && a.sig == b.sig; })) {
    available_list_ = std::move(available);
    available_row_ = -1;
  }
}
