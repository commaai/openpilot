#include "tools/cabana/ui/app.h"

#include <algorithm>
#include <filesystem>
#include <vector>

#include "imgui_internal.h"
#include "tools/cabana/commands.h"
#include "tools/cabana/dbc/dbcmanager.h"
#include "tools/cabana/settings.h"
#include "tools/cabana/streams/replaystream.h"
#include "tools/cabana/utils/util.h"

namespace fs = std::filesystem;

namespace {

constexpr const char *WIN_MESSAGES = "Messages";
constexpr const char *WIN_DETAIL = "Detail";
constexpr const char *WIN_VIDEO = "Video";
constexpr const char *WIN_CHARTS = "Charts";

int resizeCallback(ImGuiInputTextCallbackData *data) {
  if (data->EventFlag == ImGuiInputTextFlags_CallbackResize) {
    auto *s = static_cast<std::string *>(data->UserData);
    s->resize(data->BufTextLen);
    data->Buf = s->data();
  }
  return 0;
}

bool inputText(const char *label, std::string *s, const char *hint = "") {
  return ImGui::InputTextWithHint(label, hint, s->data(), s->capacity() + 1,
                                  ImGuiInputTextFlags_CallbackResize, resizeCallback, s);
}

std::vector<std::string> opendbcNames() {
  std::vector<std::string> names;
  std::error_code ec;
  for (const auto &entry : fs::directory_iterator(OPENDBC_FILE_PATH, ec)) {
    if (entry.is_regular_file() && entry.path().extension() == ".dbc") names.push_back(entry.path().filename().string());
  }
  std::sort(names.begin(), names.end());
  return names;
}

void loadDBC(App *app, const std::string &path) {
  std::string err;
  if (dbc()->open(SOURCE_ALL, path, &err)) {
    showStatus(app, "DBC file [" + path + "] loaded");
  } else {
    app->ui.error_text = path + ": " + err;
  }
}

void drawMenuBar(App *app) {
  UiState &ui = app->ui;
  if (!ImGui::BeginMainMenuBar()) return;
  const bool has_stream = app->has_stream();

  if (ImGui::BeginMenu("File")) {
    if (ImGui::MenuItem("Open Stream...")) ui.open_stream_selector = true;
    if (ImGui::MenuItem("Close stream", nullptr, false, has_stream)) closeStream(app);
    ImGui::MenuItem("Export to CSV...", nullptr, false, false);
    ImGui::Separator();
    if (ImGui::MenuItem("New DBC File", "Ctrl+N")) {
      dbc()->closeAll();
      dbc()->open(SOURCE_ALL, "untitled", "");
    }
    ImGui::MenuItem("Open DBC File...", "Ctrl+O", false, false);
    if (ImGui::BeginMenu("Open Recent", !settings.recent_files.empty())) {
      for (const auto &f : settings.recent_files) {
        if (ImGui::MenuItem(f.c_str())) loadDBC(app, f);
      }
      ImGui::EndMenu();
    }
    ImGui::Separator();
    if (ImGui::BeginMenu("Load DBC from commaai/opendbc")) {
      static const std::vector<std::string> names = opendbcNames();
      for (const auto &name : names) {
        if (ImGui::MenuItem(name.c_str())) loadDBC(app, (fs::path(OPENDBC_FILE_PATH) / name).string());
      }
      ImGui::EndMenu();
    }
    if (ImGui::MenuItem("Load DBC From Clipboard")) {
      std::string text;
      if (utils::getClipboardText(&text)) {
        std::string err;
        if (!dbc()->open(SOURCE_ALL, "clipboard", text, &err)) ui.error_text = err;
      }
    }
    ImGui::Separator();
    ImGui::MenuItem("Save DBC...", "Ctrl+S", false, false);
    ImGui::MenuItem("Save DBC As...", "Ctrl+Shift+S", false, false);
    ImGui::MenuItem("Copy DBC To Clipboard", nullptr, false, false);
    ImGui::Separator();
    if (ImGui::MenuItem("Settings...")) ui.open_settings = true;
    ImGui::Separator();
    if (ImGui::MenuItem("Exit", "Ctrl+Q")) ui.request_close = true;
    ImGui::EndMenu();
  }

  if (ImGui::BeginMenu("Edit")) {
    UndoStack *undo = UndoStack::instance();
    const std::string undo_text = "Undo " + undo->undoText();
    const std::string redo_text = "Redo " + undo->redoText();
    if (ImGui::MenuItem(undo_text.c_str(), "Ctrl+Z", false, undo->canUndo())) undo->undo();
    if (ImGui::MenuItem(redo_text.c_str(), "Ctrl+Shift+Z", false, undo->canRedo())) undo->redo();
    ImGui::EndMenu();
  }

  if (ImGui::BeginMenu("View")) {
    ImGui::MenuItem(WIN_MESSAGES, nullptr, &ui.show_messages);
    ImGui::MenuItem(WIN_DETAIL, nullptr, &ui.show_detail);
    ImGui::MenuItem(WIN_VIDEO, nullptr, &ui.show_video);
    ImGui::MenuItem(WIN_CHARTS, nullptr, &ui.show_charts);
    ImGui::Separator();
    ImGui::MenuItem("FPS overlay", nullptr, &ui.show_fps);
    if (ImGui::MenuItem("Reset Window Layout")) {
      ui.show_messages = ui.show_detail = ui.show_video = ui.show_charts = true;
      ui.reset_layout = true;
    }
    ImGui::EndMenu();
  }

  if (ImGui::BeginMenu("Tools", has_stream)) {
    ImGui::MenuItem("Find Similar Bits", nullptr, false, false);
    ImGui::MenuItem("Find Signal", nullptr, false, false);
    ImGui::EndMenu();
  }

  if (ImGui::BeginMenu("Help")) {
    ImGui::MenuItem("Help", "F1", false, false);
    ImGui::EndMenu();
  }
  ImGui::EndMainMenuBar();
}

void drawMessagesPanel(App *app) {
  UiState &ui = app->ui;
  if (!ImGui::Begin(WIN_MESSAGES, &ui.show_messages)) {
    ImGui::End();
    return;
  }
  const auto &msgs = can->lastMessages();
  std::vector<const MessageId *> ids;
  ids.reserve(msgs.size());
  for (const auto &[id, _] : msgs) ids.push_back(&id);
  std::sort(ids.begin(), ids.end(), [](const MessageId *a, const MessageId *b) { return *a < *b; });

  const ImGuiTableFlags flags = ImGuiTableFlags_RowBg | ImGuiTableFlags_Borders | ImGuiTableFlags_ScrollY |
                                ImGuiTableFlags_Resizable | ImGuiTableFlags_SizingStretchProp;
  if (ImGui::BeginTable("messages", 6, flags)) {
    ImGui::TableSetupScrollFreeze(0, 1);
    ImGui::TableSetupColumn("Name", ImGuiTableColumnFlags_WidthStretch, 2.0f);
    ImGui::TableSetupColumn("Bus", ImGuiTableColumnFlags_WidthFixed, 40.0f);
    ImGui::TableSetupColumn("ID", ImGuiTableColumnFlags_WidthFixed, 60.0f);
    ImGui::TableSetupColumn("Freq", ImGuiTableColumnFlags_WidthFixed, 60.0f);
    ImGui::TableSetupColumn("Count", ImGuiTableColumnFlags_WidthFixed, 70.0f);
    ImGui::TableSetupColumn("Bytes", ImGuiTableColumnFlags_WidthStretch, 3.0f);
    ImGui::TableHeadersRow();

    ImGuiListClipper clipper;
    clipper.Begin(static_cast<int>(ids.size()));
    while (clipper.Step()) {
      for (int row = clipper.DisplayStart; row < clipper.DisplayEnd; ++row) {
        const MessageId &id = *ids[row];
        const CanData &data = msgs.at(id);
        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        ImGui::PushID(row);
        const bool selected = ui.selected_id == id;
        if (ImGui::Selectable(msgName(id).c_str(), selected, ImGuiSelectableFlags_SpanAllColumns)) ui.selected_id = id;
        ImGui::PopID();
        ImGui::TableSetColumnIndex(1);
        ImGui::Text("%u", id.source);
        ImGui::TableSetColumnIndex(2);
        ImGui::Text("%X", id.address);
        ImGui::TableSetColumnIndex(3);
        ImGui::Text("%.1f", data.freq);
        ImGui::TableSetColumnIndex(4);
        ImGui::Text("%u", data.count);
        ImGui::TableSetColumnIndex(5);
        pushMonoFont();
        std::string hex;
        hex.reserve(data.dat.size() * 3);
        for (uint8_t b : data.dat) {
          char buf[4];
          snprintf(buf, sizeof(buf), "%02X ", b);
          hex += buf;
        }
        ImGui::TextUnformatted(hex.c_str());
        popMonoFont();
      }
    }
    ImGui::EndTable();
  }
  ImGui::End();
}

void drawDetailPanel(App *app) {
  UiState &ui = app->ui;
  if (!ImGui::Begin(WIN_DETAIL, &ui.show_detail)) {
    ImGui::End();
    return;
  }
  if (ui.selected_id.address == 0 && ui.selected_id.source == 0) {
    ImGui::TextDisabled("Select a message");
  } else {
    pushBoldFont();
    ImGui::Text("%s (%s)", msgName(ui.selected_id).c_str(), ui.selected_id.toString().c_str());
    popBoldFont();
    if (ImGui::BeginTabBar("detail_tabs")) {
      if (ImGui::BeginTabItem("Binary")) {
        ImGui::TextDisabled("binary view: not ported yet");
        ImGui::EndTabItem();
      }
      if (ImGui::BeginTabItem("Signals")) {
        ImGui::TextDisabled("signal view: not ported yet");
        ImGui::EndTabItem();
      }
      if (ImGui::BeginTabItem("Logs")) {
        ImGui::TextDisabled("history log: not ported yet");
        ImGui::EndTabItem();
      }
      ImGui::EndTabBar();
    }
  }
  ImGui::End();
}

void drawVideoPanel(App *app) {
  UiState &ui = app->ui;
  if (!ImGui::Begin(WIN_VIDEO, &ui.show_video)) {
    ImGui::End();
    return;
  }
  if (!app->has_stream()) {
    ImGui::TextDisabled("No stream");
  } else if (can->liveStreaming()) {
    ImGui::Text("%s", can->routeName().c_str());
    ImGui::Text("%.2f s", can->currentSec());
  } else {
    ImGui::Text("%s", can->routeName().c_str());
    ImGui::TextDisabled("camera view: not ported yet");
    const bool paused = can->isPaused();
    if (ImGui::Button(paused ? "Play" : "Pause")) can->pause(!paused);
    ImGui::SameLine();
    float sec = static_cast<float>(can->currentSec());
    ImGui::SetNextItemWidth(-1.0f);
    if (ImGui::SliderFloat("##seek", &sec, static_cast<float>(can->minSeconds()), static_cast<float>(can->maxSeconds()), "%.2f s")) {
      can->seekTo(sec);
    }
  }
  ImGui::End();
}

void drawChartsPanel(App *app) {
  UiState &ui = app->ui;
  if (ImGui::Begin(WIN_CHARTS, &ui.show_charts)) {
    ImGui::TextDisabled("charts: not ported yet");
  }
  ImGui::End();
}

void drawStatusBar(App *app, float height) {
  UiState &ui = app->ui;
  ImGui::PushStyleColor(ImGuiCol_ChildBg, ImGui::GetStyle().Colors[ImGuiCol_MenuBarBg]);
  ImGui::BeginChild("status_bar", ImVec2(0, height), ImGuiChildFlags_None, ImGuiWindowFlags_NoScrollbar);
  ImGui::AlignTextToFramePadding();
  if (!ui.status_text.empty() && (ui.status_until == 0 || ImGui::GetTime() < ui.status_until)) {
    ImGui::TextUnformatted(ui.status_text.c_str());
  } else {
    ui.status_text.clear();
    ImGui::TextUnformatted(app->has_stream() ? can->routeName().c_str() : "No Stream");
  }
  if (app->has_stream()) {
    const std::string right = std::to_string(can->lastMessages().size()) + " messages";
    ImGui::SameLine(ImGui::GetContentRegionAvail().x - ImGui::CalcTextSize(right.c_str()).x);
    ImGui::TextUnformatted(right.c_str());
  }
  ImGui::EndChild();
  ImGui::PopStyleColor();
}

void drawDockspace(App *app) {
  UiState &ui = app->ui;
  const ImGuiViewport *viewport = ImGui::GetMainViewport();
  ImGui::SetNextWindowPos(viewport->WorkPos);
  ImGui::SetNextWindowSize(viewport->WorkSize);
  ImGui::SetNextWindowViewport(viewport->ID);
  ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.0f);
  ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
  ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 0.0f));
  const ImGuiWindowFlags flags = ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoResize |
                                 ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoBringToFrontOnFocus | ImGuiWindowFlags_NoNavFocus |
                                 ImGuiWindowFlags_NoDocking | ImGuiWindowFlags_NoBackground;
  ImGui::Begin("##host", nullptr, flags);
  ImGui::PopStyleVar(3);

  const float status_height = ImGui::GetFrameHeight();
  const ImVec2 dock_size(ImGui::GetContentRegionAvail().x, ImGui::GetContentRegionAvail().y - status_height);
  const ImGuiID dock_id = ImGui::GetID("cabana_dockspace");
  if (ui.reset_layout) {
    ImGui::DockBuilderRemoveNode(dock_id);
    ImGui::DockBuilderAddNode(dock_id, ImGuiDockNodeFlags_DockSpace);
    ImGui::DockBuilderSetNodeSize(dock_id, dock_size);
    ImGuiID center = dock_id, left = 0, right = 0, bottom = 0;
    ImGui::DockBuilderSplitNode(center, ImGuiDir_Left, 0.28f, &left, &center);
    ImGui::DockBuilderSplitNode(center, ImGuiDir_Right, 0.35f, &right, &center);
    ImGui::DockBuilderSplitNode(center, ImGuiDir_Down, 0.35f, &bottom, &center);
    ImGui::DockBuilderDockWindow(WIN_MESSAGES, left);
    ImGui::DockBuilderDockWindow(WIN_VIDEO, right);
    ImGui::DockBuilderDockWindow(WIN_CHARTS, bottom);
    ImGui::DockBuilderDockWindow(WIN_DETAIL, center);
    ImGui::DockBuilderFinish(dock_id);
    ui.reset_layout = false;
  }
  ImGui::DockSpace(dock_id, dock_size);
  drawStatusBar(app, status_height);
  ImGui::End();
}

void drawErrorPopup(App *app);

void drawStreamSelector(App *app) {
  UiState &ui = app->ui;
  if (ui.open_stream_selector) {
    ImGui::OpenPopup("Open Stream");
    ui.open_stream_selector = false;
  }
  if (!ImGui::BeginPopupModal("Open Stream", nullptr, ImGuiWindowFlags_AlwaysAutoResize)) return;
  ImGui::TextDisabled("replay only; panda / socketcan / device tabs: not ported yet");
  ImGui::PushItemWidth(420.0f);
  inputText("Route", &app->route_input, "dongle_id/route_id or local path");
  inputText("Data dir", &app->data_dir_input, "optional local directory with routes");
  ImGui::PopItemWidth();
  ImGui::Separator();
  auto open = [&](const std::string &route) {
    Options opts;
    opts.route = route;
    opts.data_dir = app->data_dir_input;
    std::string err;
    auto stream = createStream(opts, &err);
    if (stream) {
      startStream(app, std::move(stream), "");
      ImGui::CloseCurrentPopup();
    } else {
      ui.error_text = err.empty() ? "no route given" : err;
    }
  };
  if (ImGui::Button("Open")) open(app->route_input);
  ImGui::SameLine();
  if (ImGui::Button("Demo route")) open(DEMO_ROUTE);
  ImGui::SameLine();
  if (ImGui::Button("Cancel")) ImGui::CloseCurrentPopup();
  drawErrorPopup(app);  // nested, so the error does not replace this modal
  ImGui::EndPopup();
}

void drawSettings(App *app) {
  UiState &ui = app->ui;
  if (ui.open_settings) {
    ui.settings_draft = settings;
    ImGui::OpenPopup("Settings");
    ui.open_settings = false;
  }
  if (!ImGui::BeginPopupModal("Settings", nullptr, ImGuiWindowFlags_AlwaysAutoResize)) return;
  CabanaSettingsState &draft = ui.settings_draft;
  ImGui::SliderInt("FPS", &draft.fps, 10, 100);
  ImGui::SliderInt("Max cached minutes", &draft.max_cached_minutes, 5, 60);
  ImGui::Checkbox("Absolute time", &draft.absolute_time);
  ImGui::Checkbox("Suppress defined signals", &draft.suppress_defined_signals);
  ImGui::Checkbox("Log live stream", &draft.log_livestream);
  ImGui::TextDisabled("theme / drag direction / log path: not ported yet");
  ImGui::Separator();
  if (ImGui::Button("OK")) {
    static_cast<CabanaSettingsState &>(settings) = draft;
    settings.save();
    settings.changed();
    ImGui::CloseCurrentPopup();
  }
  ImGui::SameLine();
  if (ImGui::Button("Cancel")) ImGui::CloseCurrentPopup();
  drawErrorPopup(app);
  ImGui::EndPopup();
}

void drawErrorPopup(App *app) {
  UiState &ui = app->ui;
  if (!ui.error_text.empty() && !ImGui::IsPopupOpen("Error")) ImGui::OpenPopup("Error");
  if (!ImGui::BeginPopupModal("Error", nullptr, ImGuiWindowFlags_AlwaysAutoResize)) return;
  ImGui::PushTextWrapPos(ImGui::GetCursorPos().x + 480.0f);
  ImGui::TextWrapped("%s", ui.error_text.c_str());
  ImGui::PopTextWrapPos();
  if (ImGui::Button("OK")) {
    ui.error_text.clear();
    ImGui::CloseCurrentPopup();
  }
  ImGui::EndPopup();
}

void drawFpsOverlay(const UiState &ui) {
  if (!ui.show_fps) return;
  const ImGuiViewport *viewport = ImGui::GetMainViewport();
  char label[32];
  snprintf(label, sizeof(label), "%.1f fps", ImGui::GetIO().Framerate);
  const ImVec2 size = ImGui::CalcTextSize(label);
  const ImVec2 pos(viewport->WorkPos.x + viewport->WorkSize.x - size.x - 20.0f, viewport->WorkPos.y + 10.0f);
  ImDrawList *dl = ImGui::GetForegroundDrawList();
  dl->AddRectFilled(ImVec2(pos.x - 6, pos.y - 4), ImVec2(pos.x + size.x + 6, pos.y + size.y + 4), IM_COL32(0, 0, 0, 160), 4.0f);
  dl->AddText(pos, IM_COL32_WHITE, label);
}

void handleShortcuts(App *app) {
  const ImGuiIO &io = ImGui::GetIO();
  if (io.WantTextInput) return;
  const bool ctrl = io.KeyCtrl || io.KeySuper;
  if (ctrl && ImGui::IsKeyPressed(ImGuiKey_Q, false)) app->ui.request_close = true;
  if (ctrl && ImGui::IsKeyPressed(ImGuiKey_Z, false)) {
    if (io.KeyShift) UndoStack::instance()->redo();
    else UndoStack::instance()->undo();
  }
  if (app->has_stream() && !can->liveStreaming() && ImGui::IsKeyPressed(ImGuiKey_Space, false)) can->pause(!can->isPaused());
}

}  // namespace

void showStatus(App *app, const std::string &text, double seconds) {
  app->ui.status_text = text;
  app->ui.status_until = seconds > 0 ? ImGui::GetTime() + seconds : 0;
}

void drawFrame(App *app) {
  UiState &ui = app->ui;
  handleShortcuts(app);
  drawMenuBar(app);
  drawDockspace(app);
  if (ui.show_messages) drawMessagesPanel(app);
  if (ui.show_detail) drawDetailPanel(app);
  if (ui.show_video) drawVideoPanel(app);
  if (ui.show_charts) drawChartsPanel(app);
  drawStreamSelector(app);
  drawSettings(app);
  drawErrorPopup(app);
  drawFpsOverlay(ui);
}
