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
  assert(loggy::pane_type_count() >= 14);
  assert(loggy::pane_type("plot") != nullptr);
  assert(loggy::pane_type("messages") != nullptr);
  assert(loggy::pane_type("find_signal") != nullptr);
  assert(loggy::pane_type("find_bits") != nullptr);
  assert(loggy::pane_type("computed") != nullptr);

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
  assert(workspace.tabs.size() == 2);
  assert(workspace.tabs[1].panes.size() == 1);
  assert(workspace.tabs[1].panes[0].type == "computed");
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
  assert(jot_file.tabs.size() == 2);
  assert(count_panes_of_type(jot_file, "browser") == 1);
  assert(count_panes_of_type(jot_file, "plot") == 1);
  assert(count_panes_of_type(jot_file, "logs") == 1);
  assert(count_panes_of_type(jot_file, "map") == 1);
  assert(count_panes_of_type(jot_file, "computed") == 1);
  for (const loggy::PaneInstance &pane : jot_file.tabs[0].panes) {
    if (pane.type != "plot") continue;
    const json11::Json plot_state = parse_json_or_die(pane.state_json);
    assert(plot_state["series"].array_items().size() == 2);
  }

  loggy::RouteSelection route_selection =
    loggy::parse_route_selection("5beb9b58bd12b691/0000010a--a51155e496/2:4/q");
  assert(route_selection.dongle_id == "5beb9b58bd12b691");
  assert(route_selection.timestamp == "0000010a--a51155e496");
  assert(route_selection.slice_explicit);
  assert(route_selection.begin_segment == 2);
  assert(route_selection.end_segment == 4);
  assert(route_selection.selector == loggy::LogSelector::QLog);
  assert(route_selection.selector_explicit);
  assert(loggy::route_selection_display_slice(route_selection) == "2:4");
  assert(loggy::route_selection_full_spec(route_selection) == "5beb9b58bd12b691/0000010a--a51155e496/2:4/q");
  assert(loggy::route_useradmin_url(route_selection) ==
         "https://useradmin.comma.ai/?onebox=5beb9b58bd12b691%7C0000010a--a51155e496");
  assert(loggy::route_connect_url(route_selection) ==
         "https://connect.comma.ai/5beb9b58bd12b691/0000010a--a51155e496");
  auto maybe_route_selection = loggy::route_selection_from_text("5beb9b58bd12b691/0000010a--a51155e496/1:3/r");
  assert(maybe_route_selection.has_value());
  route_selection = *maybe_route_selection;
  assert(route_selection.begin_segment == 1);
  assert(route_selection.end_segment == 3);
  assert(route_selection.selector == loggy::LogSelector::RLog);
  assert(route_selection.selector_explicit);
  maybe_route_selection = loggy::route_selection_from_text("5beb9b58bd12b691/0000010a--a51155e496/2:/q");
  assert(maybe_route_selection.has_value());
  route_selection = *maybe_route_selection;
  assert(route_selection.begin_segment == 2);
  assert(route_selection.end_segment == -1);
  assert(route_selection.selector == loggy::LogSelector::QLog);
  assert(route_selection.selector_explicit);
  maybe_route_selection = loggy::route_selection_from_text("  5beb9b58bd12b691/0000010a--a51155e496  ");
  assert(maybe_route_selection.has_value());
  maybe_route_selection = loggy::route_selection_from_text("5beb9b58bd12b691/0000010a--a51155e496/-1:2");
  assert(!maybe_route_selection.has_value());
  assert(!loggy::route_selection_from_text("bad-route-name").has_value());
  maybe_route_selection = loggy::route_selection_from_text("5beb9b58bd12b691/0000010a--a51155e496");
  assert(maybe_route_selection.has_value());
  route_selection = *maybe_route_selection;
  assert(route_selection.begin_segment == 0);
  assert(route_selection.end_segment == -1);
  const auto periods = loggy::route_browser_periods();
  assert(periods.size() == 5);
  assert(std::string(periods[0].label) == "Last week");
  assert(periods[0].days == 7);
  assert(periods[4].days == -1);
  assert(loggy::route_browser_device_routes_url("5beb9b58bd12b691", 1700000000000ULL, 1700003600000ULL, false) ==
         "https://api.commadotai.com/v1/devices/5beb9b58bd12b691/routes_segments?start_=1700000000000&end=1700003600000");
  assert(loggy::route_browser_device_routes_url("5beb9b58bd12b691", 0, 0, true) ==
         "https://api.commadotai.com/v1/devices/5beb9b58bd12b691/routes_/preserved");
  assert(loggy::route_browser_route_files_url("5beb9b58bd12b691/0000010a--a51155e496") ==
         "https://api.commadotai.com/v1/route/5beb9b58bd12b691/0000010a--a51155e496/files");
  assert(loggy::route_browser_route_label(1735689600.0, 1735706400.0) == "Wed Jan  1 00:00:00 2025    280min");
  auto route_slice = loggy::parse_route_slice_spec("7:");
  assert(route_slice.has_value());
  assert(route_slice->first == 7);
  assert(route_slice->second == -1);
  assert(!loggy::parse_route_slice_spec("9:3").has_value());
  const std::string route_list_json = R"([{"start_time":"2024-12-31T23:59:59.000Z","end_time":"2025-01-01T00:15:59.000Z","fullname":"A/B"}])";
  const auto parsed_route_entries = loggy::parse_route_browser_routes(route_list_json, true);
  assert(parsed_route_entries.second.empty());
  assert(parsed_route_entries.first.size() == 1);
  assert(parsed_route_entries.first[0].label == loggy::route_browser_route_label(1735689599.0, 1735690559.0));
  assert(parsed_route_entries.first[0].fullname == "A/B");
  const std::string route_list_nonpreserved_json = R"([{"start_time_utc_millis":1700000000000,"end_time_utc_millis":1700003600000,"fullname":"E/F"}])";
  const auto route_entries_nonpreserved = loggy::parse_route_browser_routes(route_list_nonpreserved_json, false);
  assert(route_entries_nonpreserved.second.empty());
  assert(route_entries_nonpreserved.first.size() == 1);
  assert(!route_entries_nonpreserved.first[0].label.empty());
  assert(route_entries_nonpreserved.first[0].label == loggy::route_browser_route_label(1700000000.0, 1700003600.0));

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
  assert(history.can_undo());
  assert(history.undo() != nullptr);
  assert(history.can_redo());

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
  settings.dbc_override = dbc_path.string();
  settings.map_cache_root = (temp_dir / "map-cache").string();
  std::string settings_error;
  const std::filesystem::path settings_path = temp_dir / "settings.json";
  assert(loggy::save_loggy_settings(settings, settings_path, settings_error));

  loggy::SessionConfig session_config;
  session_config.preset = "cabana";
  session_config.settings_path = settings_path.string();
  loggy::Session session(session_config);
  assert(session.settings_path == settings_path);
  assert(session.settings.recent_dbc_files.size() == 1);
  assert(session.settings.map_cache_root == (temp_dir / "map-cache").string());
  assert(session.manual_dbc_name == dbc_path.string());
  assert(session.active_dbc_name == dbc_path.string());
  assert(session.dbc_status.find("DBC override") != std::string::npos);
  assert(session.dbc.find_dbc_file(0) != nullptr);
  assert(session.dbc.find_dbc_file(0)->name() == "assigned");
  assert(session.set_manual_dbc_name("", settings_error));
  assert(settings_error.empty());
  assert(session.manual_dbc_name.empty());
  assert(session.settings.dbc_override.empty());
  assert(session.set_manual_dbc_name("honda_civic_touring_2016_can_generated", settings_error));
  assert(settings_error.empty());
  assert(session.manual_dbc_name == "honda_civic_touring_2016_can_generated");
  assert(session.active_dbc_name == "honda_civic_touring_2016_can_generated");
  assert(session.dbc.find_dbc_file(0) != nullptr);

  const std::filesystem::path settings_parent_file = temp_dir / "settings_parent_file";
  {
    std::ofstream blocker(settings_parent_file);
    blocker << "not a directory";
  }
  loggy::SessionConfig bad_settings_config;
  bad_settings_config.preset = "cabana";
  bad_settings_config.settings_path = (settings_parent_file / "settings.json").string();
  loggy::Session bad_settings_session(bad_settings_config);
  settings_error.clear();
  assert(!bad_settings_session.set_manual_dbc_name("honda_civic_touring_2016_can_generated", settings_error));
  assert(!settings_error.empty());
  assert(bad_settings_session.manual_dbc_name.empty());
  assert(bad_settings_session.settings.dbc_override.empty());
  assert(bad_settings_session.active_dbc_name.empty());
  const std::filesystem::path draft_layout_path = temp_dir / "draft_layout.json";
  loggy::Workspace base_layout = loggy::make_empty_workspace();
  base_layout.tabs[0].name = "Base";
  loggy::save_workspace_json(base_layout, draft_layout_path);
  loggy::Workspace draft_layout = loggy::make_empty_workspace();
  draft_layout.tabs[0].name = "Draft";
  loggy::split_pane(&draft_layout.tabs[0], 0, loggy::PaneSplit::Right, loggy::make_pane("plot", "Plot"));
  loggy::save_workspace_draft(draft_layout, draft_layout_path);
  loggy::WorkspaceLoadResult load_result = loggy::load_workspace_or_draft(draft_layout_path);
  const bool loaded_draft = load_result.loaded_draft;
  loggy::Workspace loaded_layout = load_result.workspace;
  assert(loaded_draft);
  assert(loaded_layout.tabs[0].name == "Draft");

  loggy::SessionConfig draft_config;
  draft_config.layout = draft_layout_path.string();
  draft_config.settings_path = (temp_dir / "draft_settings.json").string();
  loggy::Session draft_session(draft_config);
  assert(draft_session.loaded_workspace_draft);
  assert(draft_session.workspace_layout_path == std::filesystem::absolute(draft_layout_path));
  assert(draft_session.workspace.tabs[0].name == "Draft");
  assert(count_panes_of_type(draft_session.workspace, "plot") == 1);
  loggy::clear_workspace_draft(draft_layout_path);
  load_result = loggy::load_workspace_or_draft(draft_layout_path);
  assert(!load_result.loaded_draft);
  loaded_layout = load_result.workspace;
  assert(loaded_layout.tabs[0].name == "Base");

  loggy::SessionConfig preset_config;
  preset_config.preset = "jotpluggler";
  preset_config.settings_path = (temp_dir / "preset_settings.json").string();
  loggy::Session preset_session(preset_config);
  assert(count_panes_of_type(preset_session.workspace, "browser") == 1);
  assert(count_panes_of_type(preset_session.workspace, "plot") == 1);
  assert(count_panes_of_type(preset_session.workspace, "computed") == 1);
  bool saw_file_backed_plot_state = false;
  for (const loggy::PaneInstance &pane : preset_session.workspace.tabs[0].panes) {
    if (pane.type != "plot") continue;
    const json11::Json state = parse_json_or_die(pane.state_json);
    saw_file_backed_plot_state = state["series"].array_items().size() == 2;
  }
  assert(saw_file_backed_plot_state);

  const std::filesystem::path computed_layout_path = temp_dir / "computed_layout.json";
  {
    std::ofstream computed_layout(computed_layout_path);
    computed_layout << R"({
      "current_tab_index": 0,
      "tabs": [{
        "name": "Computed",
        "panes": [{
          "type": "plot",
          "title": "Computed Plot",
          "state": {
            "series": [{
              "path": "/carState/vEgo",
              "label": "scaled vEgo",
              "transform": "scale",
              "scale": 2,
              "offset": 1
            }, {
              "path": "/carState/vEgo",
              "label": "custom vEgo",
              "custom_python": {
                "linked_source": "/carState/vEgo",
                "additional_sources": [],
                "globals_code": "",
                "function_code": "return value * 3"
              }
            }]
          }
        }],
        "root": {"type": "plot", "title": "Computed Plot", "state": {"series": [{"path": "/carState/vEgo"}]}}
      }]
    })";
  }
  loggy::SessionConfig computed_config;
  computed_config.layout = computed_layout_path.string();
  computed_config.stream = true;
  computed_config.settings_path = (temp_dir / "computed_settings.json").string();
  loggy::Session computed_session(computed_config);
  assert(computed_session.computed_specs.size() == 2);
  std::string route_error;
  assert(!computed_session.restart_route("", route_error));
  assert(route_error == "route name is empty");
  assert(!computed_session.restart_route("not-a-route", route_error));
  assert(route_error == "invalid route format");
  std::string live_error;
  assert(computed_session.restart_live("localhost", 12.0, live_error));
  assert(live_error.empty());
  assert(computed_session.config.stream);
  assert(computed_session.config.stream_address == "127.0.0.1");
  assert(computed_session.config.stream_source_kind == loggy::LiveSourceKind::CerealLocal);
  assert(computed_session.config.stream_buffer_seconds == 12.0);
  assert(computed_session.live_follow);
  assert(!computed_session.toggle_live_follow());
  computed_session.set_live_follow(true);
  assert(computed_session.live_follow);
  assert(computed_session.toggle_live_paused());
  assert(computed_session.live_status().paused);
  computed_session.set_live_paused(false);
  assert(!computed_session.live_status().paused);
  assert(!computed_session.restart_live(loggy::LiveSourceKind::SocketCan, "", 5.0, live_error));
  assert(live_error == "SocketCAN device is empty");
  assert(computed_session.restart_live(loggy::LiveSourceKind::SocketCan, "vcan-test", 5.0, live_error));
  assert(live_error.empty());
  assert(computed_session.config.stream_source_kind == loggy::LiveSourceKind::SocketCan);
  assert(computed_session.config.stream_address == "vcan-test");
  assert(computed_session.restart_live(loggy::LiveSourceKind::PandaUsb, "", 5.0, live_error));
  assert(live_error.empty());
  assert(computed_session.config.stream_source_kind == loggy::LiveSourceKind::PandaUsb);
  assert(computed_session.config.stream_address.empty());
  assert(computed_session.restart_live(loggy::LiveSourceKind::PandaUsb, "panda-serial", 5.0, live_error));
  assert(live_error.empty());
  assert(computed_session.config.stream_source_kind == loggy::LiveSourceKind::PandaUsb);
  assert(computed_session.config.stream_address == "panda-serial");
  loggy::LiveSourceConfig tuned_panda_source;
  tuned_panda_source.kind = loggy::LiveSourceKind::PandaUsb;
  tuned_panda_source.address = "tuned-panda";
  tuned_panda_source.buffer_seconds = 9.0;
  tuned_panda_source.panda_buses[0] = loggy::PandaBusConfig{.can_speed_kbps = 250, .data_speed_kbps = 1000, .can_fd = true};
  tuned_panda_source.panda_buses[1] = loggy::PandaBusConfig{.can_speed_kbps = 125, .data_speed_kbps = 5000, .can_fd = false};
  tuned_panda_source.panda_buses[2] = loggy::PandaBusConfig{.can_speed_kbps = 333, .data_speed_kbps = 333, .can_fd = true};
  assert(computed_session.restart_live(tuned_panda_source, live_error));
  assert(live_error.empty());
  assert(computed_session.config.stream_source_kind == loggy::LiveSourceKind::PandaUsb);
  assert(computed_session.config.stream_address == "tuned-panda");
  assert(computed_session.config.stream_buffer_seconds == 9.0);
  assert(computed_session.config.stream_panda_buses[0].can_speed_kbps == 250);
  assert(computed_session.config.stream_panda_buses[0].data_speed_kbps == 1000);
  assert(computed_session.config.stream_panda_buses[0].can_fd);
  assert(computed_session.config.stream_panda_buses[1].can_speed_kbps == 125);
  assert(computed_session.config.stream_panda_buses[1].data_speed_kbps == 5000);
  assert(!computed_session.config.stream_panda_buses[1].can_fd);
  assert(computed_session.config.stream_panda_buses[2].can_speed_kbps == 500);
  assert(computed_session.config.stream_panda_buses[2].data_speed_kbps == 2000);
  assert(computed_session.config.stream_panda_buses[2].can_fd);
  loggy::SessionConfig remote_local_config;
  remote_local_config.settings_path = (temp_dir / "remote_local_settings.json").string();
  remote_local_config.stream_source_kind = loggy::LiveSourceKind::CerealRemote;
  remote_local_config.stream_address = "127.0.0.1";
  loggy::Session remote_local_session(remote_local_config);
  assert(remote_local_session.config.stream_source_kind == loggy::LiveSourceKind::CerealRemote);
  assert(remote_local_session.config.stream_address == "127.0.0.1");
  loggy::SessionConfig bridge_config;
  bridge_config.settings_path = (temp_dir / "bridge_settings.json").string();
  bridge_config.stream_source_kind = loggy::LiveSourceKind::DeviceBridge;
  bridge_config.stream_address = "192.168.0.10";
  loggy::Session bridge_session(bridge_config);
  assert(bridge_session.config.stream_source_kind == loggy::LiveSourceKind::DeviceBridge);
  assert(bridge_session.config.stream_address == "192.168.0.10");
  loggy::SessionConfig panda_config;
  panda_config.settings_path = (temp_dir / "panda_settings.json").string();
  panda_config.stream_source_kind = loggy::LiveSourceKind::PandaUsb;
  panda_config.stream_address = "";
  loggy::Session panda_session(panda_config);
  assert(panda_session.config.stream_source_kind == loggy::LiveSourceKind::PandaUsb);
  assert(panda_session.config.stream_address.empty());
  const std::string computed_path = computed_session.computed_specs[0].output_path;
  const std::string custom_path = computed_session.computed_specs[1].output_path;
  assert(computed_path.rfind("/computed/scaled-vego-", 0) == 0);
  assert(custom_path.rfind("/computed/custom-vego-", 0) == 0);
  const json11::Json computed_plot_state = parse_json_or_die(computed_session.workspace.tabs[0].panes[0].state_json);
  const json11::Json computed_series = computed_plot_state["series"].array_items()[0];
  const json11::Json custom_series = computed_plot_state["series"].array_items()[1];
  assert(computed_series["path"].string_value() == computed_path);
  assert(computed_series["computed_source_path"].string_value() == "/carState/vEgo");
  assert(!computed_series["transform"].is_string());
  assert(custom_series["path"].string_value() == custom_path);
  assert(custom_series["custom_python"].is_object());
  const loggy::DrainResult computed_drain = computed_session.begin_frame();
  assert(computed_drain.series_chunks >= 4);
  const loggy::SeriesView gps_view =
    computed_session.store.series_full("/gpsLocationExternal/latitude", {0.0, 60.0});
  assert(gps_view.points.size() > 10);
  const loggy::SeriesView computed_view = computed_session.store.series_full(computed_path, {0.0, 60.0});
  assert(!computed_view.points.empty());
  assert(computed_view.points[0].value == 37.0);
  const loggy::SeriesView custom_view = computed_session.store.series_full(custom_path, {0.0, 60.0});
  assert(!custom_view.points.empty());
  assert(custom_view.points[0].value == 54.0);
  assert(!computed_session.computed_statuses.empty());
  assert(computed_session.computed_statuses[0].ok);
  assert(computed_session.computed_statuses[1].ok);

  std::filesystem::remove_all(temp_dir);

  return 0;
}
