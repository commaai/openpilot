#include "tools/cabana/imgui/file_dialog.h"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

#include "imgui.h"

namespace fs = std::filesystem;

namespace {

struct Entry {
  std::string name;
  bool is_dir = false;
};

struct FileDialogState {
  bool need_open = false;  // edge flag: call ImGui::OpenPopup() next draw
  bool active = false;     // a request is open or pending
  FileDialogMode mode = FileDialogMode::Open;
  std::string title;
  std::string filter_ext;
  std::function<void(const std::string &)> on_ok;

  fs::path current_dir;
  char path_buf[1024] = {};
  char name_buf[512] = {};
  std::string selected_entry;
  std::vector<Entry> entries;
  std::string status;  // inline error/status text (e.g. "File not found")
  bool refresh_needed = true;
};

FileDialogState g_fd;

bool ends_with_ci(const std::string &s, const std::string &suffix) {
  if (suffix.empty()) return true;
  if (s.size() < suffix.size()) return false;
  return std::equal(suffix.rbegin(), suffix.rend(), s.rbegin(), [](char a, char b) {
    return std::tolower(static_cast<unsigned char>(a)) == std::tolower(static_cast<unsigned char>(b));
  });
}

std::string to_lower(std::string s) {
  std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return std::tolower(c); });
  return s;
}

void set_path_buf(const fs::path &p) {
  std::snprintf(g_fd.path_buf, sizeof(g_fd.path_buf), "%s", p.string().c_str());
}

// mirrors QDir::entryList({filter}, QDir::Files, QDir::Name): directories
// first (Qt's opendbc menu build uses QDir::Files only, but a general-purpose
// browser needs to navigate into directories too), case-insensitive name sort.
void refresh_listing() {
  g_fd.entries.clear();
  g_fd.status.clear();
  std::error_code ec;
  fs::directory_iterator it(g_fd.current_dir, fs::directory_options::skip_permission_denied, ec);
  if (ec) {
    g_fd.status = ec.message();
    g_fd.refresh_needed = false;
    return;
  }
  for (const auto &de : it) {
    std::error_code ec2;
    const bool is_dir = de.is_directory(ec2);
    std::string name = de.path().filename().string();
    if (name.empty() || name[0] == '.') continue;  // hide dotfiles/dotdirs
    if (!is_dir && !ends_with_ci(name, g_fd.filter_ext)) continue;
    g_fd.entries.push_back({name, is_dir});
  }
  std::sort(g_fd.entries.begin(), g_fd.entries.end(), [](const Entry &a, const Entry &b) {
    if (a.is_dir != b.is_dir) return a.is_dir;
    return to_lower(a.name) < to_lower(b.name);
  });
  g_fd.refresh_needed = false;
}

void navigate_to(const fs::path &dir) {
  std::error_code ec;
  fs::path canon = fs::weakly_canonical(dir, ec);
  g_fd.current_dir = ec ? dir : canon;
  set_path_buf(g_fd.current_dir);
  g_fd.selected_entry.clear();
  g_fd.refresh_needed = true;
}

void do_accept(const std::string &filename) {
  if (filename.empty()) return;
  const fs::path full = g_fd.current_dir / filename;
  if (g_fd.mode == FileDialogMode::Open) {
    std::error_code ec;
    if (!fs::is_regular_file(full, ec)) {
      g_fd.status = "File not found";
      return;
    }
  }
  auto on_ok = std::move(g_fd.on_ok);
  const std::string path_str = full.string();
  g_fd.active = false;
  g_fd.on_ok = nullptr;
  ImGui::CloseCurrentPopup();
  if (on_ok) on_ok(path_str);
}

}  // namespace

void file_dialog_open(FileDialogMode mode, const std::string &title, const std::string &start_dir,
                       const std::string &filter_ext, const std::string &default_filename,
                       std::function<void(const std::string &path)> on_ok) {
  g_fd.mode = mode;
  g_fd.title = title;
  g_fd.filter_ext = filter_ext;
  g_fd.on_ok = std::move(on_ok);
  g_fd.need_open = true;
  g_fd.active = true;
  g_fd.status.clear();

  fs::path dir = start_dir.empty() ? fs::current_path() : fs::path(start_dir);
  std::error_code ec;
  if (!fs::is_directory(dir, ec)) dir = fs::current_path();
  navigate_to(dir);

  std::snprintf(g_fd.name_buf, sizeof(g_fd.name_buf), "%s", default_filename.c_str());
}

// Fixed popup ID ("File Browser##dbc_file_dialog") shared by every call so
// re-triggering across frames (e.g. chained Save As prompts) reuses the same
// ImGui window identity; the caller-supplied `title` is rendered as a body
// line instead of the OS-level title bar text.
void file_dialog_draw() {
  constexpr const char *kPopupId = "File Browser##dbc_file_dialog";

  if (g_fd.need_open) {
    ImGui::OpenPopup(kPopupId);
    g_fd.need_open = false;
  }
  if (!g_fd.active) return;

  ImGui::SetNextWindowSize(ImVec2(640.0f, 460.0f), ImGuiCond_Appearing);
  ImGui::SetNextWindowPos(ImGui::GetMainViewport()->GetCenter(), ImGuiCond_Appearing, ImVec2(0.5f, 0.5f));
  if (!ImGui::BeginPopupModal(kPopupId, nullptr, ImGuiWindowFlags_NoSavedSettings)) return;

  ImGui::TextUnformatted(g_fd.title.c_str());
  ImGui::Separator();

  ImGui::TextUnformatted("Path:");
  ImGui::SameLine();
  ImGui::SetNextItemWidth(-1.0f);
  if (ImGui::InputText("##fd_path", g_fd.path_buf, sizeof(g_fd.path_buf), ImGuiInputTextFlags_EnterReturnsTrue)) {
    std::error_code ec;
    fs::path p(g_fd.path_buf);
    if (fs::is_directory(p, ec)) {
      navigate_to(p);
    } else {
      g_fd.status = "Not a directory";
    }
  }

  if (g_fd.refresh_needed) refresh_listing();

  const ImVec2 avail = ImGui::GetContentRegionAvail();
  const float bottom_h = ImGui::GetFrameHeightWithSpacing() * 2.0f + ImGui::GetStyle().ItemSpacing.y;
  if (ImGui::BeginChild("##fd_list", ImVec2(0.0f, avail.y - bottom_h), ImGuiChildFlags_Borders)) {
    const bool at_root = !g_fd.current_dir.has_relative_path() && g_fd.current_dir == g_fd.current_dir.root_path();
    if (!at_root) {
      if (ImGui::Selectable("..", false, ImGuiSelectableFlags_AllowDoubleClick) &&
          ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left)) {
        navigate_to(g_fd.current_dir.parent_path());
      }
    }
    for (const auto &e : g_fd.entries) {
      const bool selected = (g_fd.selected_entry == e.name);
      const std::string label = (e.is_dir ? "[dir]  " : "        ") + e.name;
      if (ImGui::Selectable(label.c_str(), selected, ImGuiSelectableFlags_AllowDoubleClick)) {
        if (e.is_dir) {
          if (ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left)) {
            navigate_to(g_fd.current_dir / e.name);
          }
        } else {
          g_fd.selected_entry = e.name;
          std::snprintf(g_fd.name_buf, sizeof(g_fd.name_buf), "%s", e.name.c_str());
          if (ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left)) {
            do_accept(e.name);
            ImGui::EndChild();
            ImGui::EndPopup();
            return;
          }
        }
      }
    }
  }
  ImGui::EndChild();

  if (!g_fd.status.empty()) {
    ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.85f, 0.3f, 0.25f, 1.0f));
    ImGui::TextUnformatted(g_fd.status.c_str());
    ImGui::PopStyleColor();
  }

  ImGui::TextUnformatted("File name:");
  ImGui::SameLine();
  ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x);
  const bool enter_pressed =
      ImGui::InputText("##fd_name", g_fd.name_buf, sizeof(g_fd.name_buf), ImGuiInputTextFlags_EnterReturnsTrue);

  const bool ok_clicked = ImGui::Button(g_fd.mode == FileDialogMode::Save ? "Save" : "Open", ImVec2(100.0f, 0.0f));
  ImGui::SameLine();
  const bool cancel_clicked =
      ImGui::Button("Cancel", ImVec2(100.0f, 0.0f)) || ImGui::IsKeyPressed(ImGuiKey_Escape, false);

  if (enter_pressed || ok_clicked) {
    do_accept(g_fd.name_buf);
  } else if (cancel_clicked) {
    g_fd.active = false;
    g_fd.on_ok = nullptr;
    ImGui::CloseCurrentPopup();
  }

  ImGui::EndPopup();
}
