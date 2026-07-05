#include "tools/loggy/backend/session.h"
#include "tools/loggy/backend/dbc/dbcmanager.h"
#include "tools/loggy/shell/settings.h"
#include "tools/loggy/shell/workspace.h"

#include "json11/json11.hpp"

#include <array>
#include <cassert>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace {

int count_panes_of_type(const loggy::Workspace &workspace, const std::string &type) {
  int count = 0;
  for (const loggy::WorkspaceTab &tab : workspace.tabs) {
    for (const loggy::PaneInstance &pane : tab.panes) {
      if (pane.type == type) ++count;
    }
  }
  return count;
}

json11::Json parse_json_or_die(const std::string &text) {
  std::string err;
  const json11::Json parsed = json11::Json::parse(text, err);
  assert(err.empty());
  return parsed;
}

}  // namespace

int main() {
  loggy::PaneRegistry registry;
  loggy::register_dummy_pane_types(registry);
  assert(registry.find("plot") != nullptr);
  assert(registry.find("messages") != nullptr);
  assert(registry.find("find_signal") != nullptr);
  assert(registry.find("find_bits") != nullptr);

  loggy::Workspace cabana = loggy::make_cabana_workspace();
  assert(cabana.tabs.size() == 3);
  assert(cabana.tabs[0].panes.size() == 4);
  assert(cabana.tabs[0].panes[0].type == "messages");
  assert(cabana.tabs[1].name == "DBC");
  assert(cabana.tabs[1].panes.size() == 1);
  assert(cabana.tabs[1].panes[0].type == "dbc");
  assert(cabana.tabs[2].name == "Analysis");
  assert(cabana.tabs[2].panes.size() == 2);
  assert(cabana.tabs[2].panes[0].type == "find_signal");
  assert(cabana.tabs[2].panes[1].type == "find_bits");
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

  const std::filesystem::path layout_dir = loggy::layouts_dir();
  const loggy::Workspace cabana_file = loggy::load_workspace_json(layout_dir / "cabana.json");
  assert(cabana_file.tabs.size() == 3);
  assert(cabana_file.tabs[0].panes.size() == 4);
  assert(count_panes_of_type(cabana_file, "messages") == 1);
  assert(count_panes_of_type(cabana_file, "binary") == 1);
  assert(count_panes_of_type(cabana_file, "dbc") == 1);
  assert(count_panes_of_type(cabana_file, "find_signal") == 1);

  const loggy::Workspace jot_file = loggy::load_workspace_json(layout_dir / "jotpluggler.json");
  assert(jot_file.tabs.size() == 1);
  assert(count_panes_of_type(jot_file, "browser") == 1);
  assert(count_panes_of_type(jot_file, "plot") == 1);
  assert(count_panes_of_type(jot_file, "logs") == 1);
  assert(count_panes_of_type(jot_file, "map") == 1);
  for (const loggy::PaneInstance &pane : jot_file.tabs[0].panes) {
    if (pane.type != "plot") continue;
    const json11::Json plot_state = parse_json_or_die(pane.state_json);
    assert(plot_state["series"].array_items().size() == 2);
  }

  const std::array<const char *, 17> jot_layouts = {{
    "CAN-bus-debug.json",
    "camera-timings.json",
    "cameras-and-map.json",
    "can-states.json",
    "controls_mismatch_debug.json",
    "driver-monitoring-debug.json",
    "gps.json",
    "gps_vs_llk.json",
    "locationd_debug.json",
    "longitudinal.json",
    "max-torque-debug.json",
    "new-layout.json",
    "system_lag_debug.json",
    "thermal_debug.json",
    "torque-controller.json",
    "tuning.json",
    "ublox-debug.json",
  }};
  for (const char *name : jot_layouts) {
    const loggy::Workspace imported = loggy::load_workspace_json(layout_dir / name);
    assert(!imported.tabs.empty());
    for (const loggy::WorkspaceTab &imported_tab : imported.tabs) {
      assert(!imported_tab.panes.empty());
      for (const loggy::PaneInstance &pane : imported_tab.panes) {
        assert(pane.type != loggy::kDefaultPaneType);
      }
    }
  }

  const loggy::Workspace longitudinal = loggy::load_workspace_json(layout_dir / "longitudinal.json");
  assert(longitudinal.tabs.size() == 1);
  assert(longitudinal.tabs[0].panes.size() == 4);
  assert(count_panes_of_type(longitudinal, "plot") == 4);
  const json11::Json longitudinal_plot = parse_json_or_die(longitudinal.tabs[0].panes[0].state_json);
  assert(longitudinal_plot["series"].array_items().size() == 4);
  assert(longitudinal_plot["series"].array_items()[0]["path"].string_value() == "/carState/aEgo");
  assert(longitudinal_plot["y_limits"]["min"].is_number());
  assert(longitudinal_plot["y_limits"]["max"].is_number());
  assert(longitudinal_plot["jotpluggler_range"]["left"].is_number());

  const loggy::Workspace cameras_and_map = loggy::load_workspace_json(layout_dir / "cameras-and-map.json");
  assert(count_panes_of_type(cameras_and_map, "map") == 1);
  assert(count_panes_of_type(cameras_and_map, "camera") == 3);
  bool saw_road = false;
  bool saw_wide = false;
  bool saw_driver = false;
  for (const loggy::PaneInstance &pane : cameras_and_map.tabs[0].panes) {
    if (pane.type != "camera") continue;
    const json11::Json camera_state = parse_json_or_die(pane.state_json);
    saw_road = saw_road || camera_state["camera_view"].string_value() == "road";
    saw_wide = saw_wide || camera_state["camera_view"].string_value() == "wide_road";
    saw_driver = saw_driver || camera_state["camera_view"].string_value() == "driver";
  }
  assert(saw_road);
  assert(saw_wide);
  assert(saw_driver);

  const loggy::Workspace torque = loggy::load_workspace_json(layout_dir / "torque-controller.json");
  bool preserved_custom = false;
  bool preserved_scale = false;
  for (const loggy::WorkspaceTab &torque_tab : torque.tabs) {
    for (const loggy::PaneInstance &pane : torque_tab.panes) {
      if (pane.type != "plot") continue;
      const json11::Json state = parse_json_or_die(pane.state_json);
      for (const json11::Json &series : state["series"].array_items()) {
        preserved_custom = preserved_custom || series["custom_python"].is_object();
        preserved_scale = preserved_scale || (series["transform"].string_value() == "scale" &&
                                             series["scale"].is_number() &&
                                             series["offset"].is_number());
      }
    }
  }
  assert(preserved_custom);
  assert(preserved_scale);

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

  loggy::SessionConfig preset_config;
  preset_config.preset = "jotpluggler";
  preset_config.settings_path = (temp_dir / "preset_settings.json").string();
  loggy::Session preset_session(preset_config);
  assert(count_panes_of_type(preset_session.workspace(), "browser") == 1);
  assert(count_panes_of_type(preset_session.workspace(), "plot") == 1);
  bool saw_file_backed_plot_state = false;
  for (const loggy::PaneInstance &pane : preset_session.workspace().tabs[0].panes) {
    if (pane.type != "plot") continue;
    const json11::Json state = parse_json_or_die(pane.state_json);
    saw_file_backed_plot_state = state["series"].array_items().size() == 2;
  }
  assert(saw_file_backed_plot_state);

  std::filesystem::remove_all(temp_dir);

  return 0;
}
