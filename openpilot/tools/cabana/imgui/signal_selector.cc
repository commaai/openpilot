// ImGui port of tools/cabana/chart/signalselector.h/.cc (SignalSelector), the
// frozen Qt reference: a two-pane "available signals for the chosen message"
// -> "selected signals" modal, reused for both ChartsWidget::newChart() and
// ChartView::manageSignals(). Selection here is done with a click (add) /
// click (remove) instead of Qt's double-click-or-button hybrid -- simpler in
// immediate mode and functionally equivalent.
#include "tools/cabana/imgui/charts_internal.h"

#include <algorithm>
#include <string>
#include <vector>

#include "tools/cabana/dbc/dbcmanager.h"

namespace {

struct MsgEntry {
  MessageId id;
  std::string label;
};

std::vector<MsgEntry> build_msg_list() {
  std::vector<MsgEntry> out;
  for (const auto &[id, _] : can->lastMessages()) {
    if (cabana::Msg *m = dbc()->msg(id); m != nullptr) {
      out.push_back({id, m->name + " (" + id.toString() + ")"});
    }
  }
  std::sort(out.begin(), out.end(), [](const MsgEntry &a, const MsgEntry &b) { return a.label < b.label; });
  return out;
}

bool is_selected(const SelectorState &sel, const MessageId &id, const cabana::Signal *sig) {
  for (const auto &[mid, s] : sel.selected) {
    if (mid == id && s == sig) return true;
  }
  return false;
}

void draw_color_swatch(const ColorRGBA &color) {
  ImVec2 p0 = ImGui::GetCursorScreenPos();
  const float sq = ImGui::GetTextLineHeight() * 0.7f;
  const float pad = (ImGui::GetTextLineHeight() - sq) * 0.5f;
  ImGui::GetWindowDrawList()->AddRectFilled(ImVec2(p0.x, p0.y + pad), ImVec2(p0.x + sq, p0.y + pad + sq), to_im_color(color));
  ImGui::Dummy(ImVec2(sq + 6.0f, ImGui::GetTextLineHeight()));
  ImGui::SameLine();
}

void apply_selector_accept() {
  SelectorState &sel = g_charts.selector;
  if (sel.target_chart_id < 0) {
    // "New Chart": mirrors ChartsWidget::newChart() -- all selected signals land in one new chart.
    if (!sel.selected.empty()) {
      ChartState &c = create_chart();
      for (auto &[id, sig] : sel.selected) add_signal_to_chart(c, id, sig);
    }
  } else if (ChartState *c = find_chart_by_id(sel.target_chart_id); c != nullptr) {
    // "Manage Chart": mirrors ChartView::manageSignals() -- add newly picked, drop unpicked.
    for (auto &[id, sig] : sel.selected) add_signal_to_chart(*c, id, sig);
    remove_signals_if(*c, [&](const SigItem &s) {
      return !std::any_of(sel.selected.begin(), sel.selected.end(),
                           [&](const auto &p) { return p.first == s.msg_id && p.second == s.sig; });
    });
    if (c->sigs.empty()) close_chart(c->id);
  }
}

}  // namespace

void draw_signal_selector_modal() {
  SelectorState &sel = g_charts.selector;
  if (sel.open_request) {
    sel.open_request = false;
    sel.visible = true;
    ImGui::OpenPopup(sel.title.c_str());
  }
  if (!sel.visible) return;

  ImGui::SetNextWindowSize(ImVec2(640.0f, 440.0f), ImGuiCond_Appearing);
  ImGui::SetNextWindowPos(ImGui::GetMainViewport()->GetCenter(), ImGuiCond_Appearing, ImVec2(0.5f, 0.5f));
  bool open = true;
  if (ImGui::BeginPopupModal(sel.title.c_str(), &open, ImGuiWindowFlags_NoSavedSettings)) {
    const std::vector<MsgEntry> msgs = build_msg_list();
    const ImVec2 pane_size(300.0f, 340.0f);

    ImGui::BeginGroup();
    ImGui::TextUnformatted("Available Signals");
    const char *preview = (sel.chosen_msg_combo >= 0 && sel.chosen_msg_combo < static_cast<int>(msgs.size()))
                               ? msgs[static_cast<size_t>(sel.chosen_msg_combo)].label.c_str()
                               : "Select a msg...";
    ImGui::SetNextItemWidth(pane_size.x);
    if (ImGui::BeginCombo("##msg_combo", preview)) {
      for (int i = 0; i < static_cast<int>(msgs.size()); ++i) {
        if (ImGui::Selectable(msgs[static_cast<size_t>(i)].label.c_str(), i == sel.chosen_msg_combo)) {
          sel.chosen_msg_combo = i;
        }
      }
      ImGui::EndCombo();
    }
    if (ImGui::BeginChild("##avail_list", pane_size, true)) {
      if (sel.chosen_msg_combo >= 0 && sel.chosen_msg_combo < static_cast<int>(msgs.size())) {
        const MessageId mid = msgs[static_cast<size_t>(sel.chosen_msg_combo)].id;
        if (cabana::Msg *m = dbc()->msg(mid); m != nullptr) {
          for (cabana::Signal *s : m->getSignals()) {
            if (is_selected(sel, mid, s)) continue;
            ImGui::PushID(s);
            draw_color_swatch(s->color);
            if (ImGui::Selectable(s->name.c_str())) sel.selected.emplace_back(mid, s);
            ImGui::PopID();
          }
        }
      }
    }
    ImGui::EndChild();
    ImGui::EndGroup();

    ImGui::SameLine();
    ImGui::BeginGroup();
    ImGui::TextUnformatted("Selected Signals");
    if (ImGui::BeginChild("##selected_list", pane_size, true)) {
      int remove_idx = -1;
      for (int i = 0; i < static_cast<int>(sel.selected.size()); ++i) {
        const auto &[mid, s] = sel.selected[static_cast<size_t>(i)];
        ImGui::PushID(i);
        draw_color_swatch(s->color);
        const std::string label = s->name + "   " + msgName(mid) + " " + mid.toString();
        if (ImGui::Selectable(label.c_str())) remove_idx = i;
        ImGui::PopID();
      }
      if (remove_idx >= 0) sel.selected.erase(sel.selected.begin() + remove_idx);
    }
    ImGui::EndChild();
    ImGui::EndGroup();

    ImGui::Spacing();
    if (ImGui::Button("OK", ImVec2(80.0f, 0.0f))) {
      apply_selector_accept();
      sel.visible = false;
      ImGui::CloseCurrentPopup();
    }
    ImGui::SameLine();
    if (ImGui::Button("Cancel", ImVec2(80.0f, 0.0f)) || !open) {
      sel.visible = false;
      ImGui::CloseCurrentPopup();
    }
    ImGui::EndPopup();
  } else if (sel.visible) {
    // Closed via Escape/click-outside without hitting OK/Cancel -- mirrors QDialog::reject().
    sel.visible = false;
  }
}
