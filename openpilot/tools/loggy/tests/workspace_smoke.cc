#include "tools/loggy/backend/session.h"
#include "tools/loggy/backend/dbc/dbcmanager.h"
#include "tools/loggy/shell/settings.h"
#include "tools/loggy/shell/workspace.h"

#include <cassert>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <string>

int main() {
  loggy::PaneRegistry registry;
  loggy::register_dummy_pane_types(registry);
  assert(registry.find("plot") != nullptr);
  assert(registry.find("messages") != nullptr);

  loggy::Workspace cabana = loggy::make_cabana_workspace();
  assert(cabana.tabs.size() == 2);
  assert(cabana.tabs[0].panes.size() == 4);
  assert(cabana.tabs[0].panes[0].type == "messages");
  assert(cabana.tabs[1].name == "DBC");
  assert(cabana.tabs[1].panes.size() == 1);
  assert(cabana.tabs[1].panes[0].type == "dbc");
  const loggy::WorkspaceNode &cabana_root = cabana.tabs[0].root;
  assert(!cabana_root.is_pane);
  assert(cabana_root.orientation == loggy::SplitOrientation::Horizontal);
  assert(cabana_root.children.size() == 2);
  assert(!cabana_root.children[0].is_pane);
  assert(cabana_root.children[0].orientation == loggy::SplitOrientation::Vertical);
  assert(cabana_root.children[0].children.size() == 3);
  assert(cabana_root.children[0].children[0].pane_index == 0);
  assert(cabana_root.children[0].children[1].pane_index == 3);
  assert(cabana_root.children[0].children[2].pane_index == 2);
  assert(cabana_root.children[1].pane_index == 1);

  loggy::Workspace workspace = loggy::make_jotpluggler_workspace();
  loggy::WorkspaceTab *tab = loggy::active_tab(&workspace);
  assert(tab != nullptr);
  assert(tab->panes.size() == 4);

  const int camera_index = loggy::add_pane(tab, loggy::make_pane("camera", "Camera"), 0, loggy::PaneSplit::Right);
  assert(camera_index == 4);
  assert(tab->panes.size() == 5);

  assert(loggy::move_pane(tab, camera_index, 1, loggy::PaneSplit::Bottom));
  assert(tab->panes.size() == 5);
  assert(loggy::close_pane(tab, 2));
  assert(tab->panes.size() == 4);

  const std::string encoded = loggy::workspace_to_json(workspace);
  loggy::Workspace decoded = loggy::workspace_from_json(encoded);
  assert(decoded.tabs.size() == workspace.tabs.size());
  assert(decoded.tabs[0].panes.size() == workspace.tabs[0].panes.size());

  loggy::WorkspaceHistory history;
  history.reset(workspace);
  loggy::add_tab(&workspace, "Review");
  history.push(workspace);
  assert(history.canUndo());
  assert(history.undo() != nullptr);
  assert(history.canRedo());

  const auto now = std::chrono::steady_clock::now().time_since_epoch().count();
  const std::filesystem::path temp_dir = std::filesystem::temp_directory_path() / ("loggy_workspace_smoke_" + std::to_string(now));
  std::filesystem::create_directories(temp_dir);
  const std::filesystem::path dbc_path = temp_dir / "assigned.dbc";
  {
    std::ofstream dbc_file(dbc_path);
    dbc_file << R"(VERSION ""
BO_ 291 TEST_MSG: 1 XXX
 SG_ sig : 0|8@1+ (1,0) [0|255] "" XXX
)";
  }
  loggy::LoggySettings settings;
  loggy::remember_recent_dbc_file(&settings, dbc_path.string());
  loggy::set_dbc_assignment(&settings, "all", dbc_path.string());
  std::string settings_error;
  const std::filesystem::path settings_path = temp_dir / "settings.json";
  assert(loggy::save_loggy_settings(settings, settings_path, &settings_error));

  loggy::dbc()->closeAll();
  loggy::SessionConfig session_config;
  session_config.preset = "cabana";
  session_config.settings_path = settings_path.string();
  loggy::Session session(session_config);
  assert(session.settings_path() == settings_path);
  assert(session.settings().recent_dbc_files.size() == 1);
  assert(loggy::dbc()->findDBCFile(0) != nullptr);
  assert(loggy::dbc()->findDBCFile(0)->name() == "assigned");
  loggy::dbc()->closeAll();
  std::filesystem::remove_all(temp_dir);

  return 0;
}
