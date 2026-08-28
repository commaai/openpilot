#include "tools/cabana/ui/dialogs/filedialog.h"

#include <algorithm>
#include <system_error>

#include "imgui.h"
#include "tools/cabana/ui/dialogs/messagebox.h"
#include "tools/cabana/ui/imgui_util.h"

namespace fs = std::filesystem;

namespace FileDialog {

namespace {

enum class Mode { OpenFile, SaveFile, Directory };

struct State {
  bool active = false;
  bool open = false;
  Mode mode = Mode::OpenFile;
  std::string title;
  std::string extension;
  fs::path dir;
  std::string dir_input;
  std::string filename;
  std::vector<fs::directory_entry> entries;
  Callback callback;
};

State g_state;

void listDir() {
  State &s = g_state;
  s.entries.clear();
  std::error_code ec;
  for (const auto &entry : fs::directory_iterator(s.dir, ec)) {
    const std::string name = entry.path().filename().string();
    if (name.empty() || name[0] == '.') continue;
    const bool is_dir = entry.is_directory(ec);
    if (!is_dir && s.mode == Mode::Directory) continue;
    if (!is_dir && !s.extension.empty() && entry.path().extension() != s.extension) continue;
    s.entries.push_back(entry);
  }
  std::sort(s.entries.begin(), s.entries.end(), [](const auto &a, const auto &b) {
    std::error_code sort_ec;
    const bool da = a.is_directory(sort_ec), db = b.is_directory(sort_ec);
    return da != db ? da : a.path().filename() < b.path().filename();
  });
  s.dir_input = s.dir.string();
}

void setDir(const fs::path &dir) {
  std::error_code ec;
  fs::path d = fs::is_directory(dir, ec) ? fs::absolute(dir, ec) : fs::current_path(ec);
  g_state.dir = d.lexically_normal();
  listDir();
}

void start(Mode mode, const std::string &title, const fs::path &dir, const std::string &filename,
           const std::string &extension, Callback cb) {
  State &s = g_state;
  s = State{};
  s.active = true;
  s.mode = mode;
  s.title = title;
  s.extension = extension;
  s.filename = filename;
  s.callback = std::move(cb);
  setDir(dir);
}

void finish(const std::string &path) {
  Callback cb = std::move(g_state.callback);
  g_state = State{};
  if (cb) cb(path);
}

void accept(const fs::path &path) {
  if (g_state.mode == Mode::SaveFile) {
    std::error_code ec;
    if (fs::exists(path, ec)) {
      const std::string name = path.filename().string();
      MessageBox::question(g_state.title, name + " already exists.\nDo you want to replace it?", [path](bool ok) {
        if (ok) finish(path.string());
      });
      return;
    }
  }
  finish(path.string());
}

}  // namespace

void getOpenFileName(const std::string &title, const std::string &dir, const std::string &extension, Callback cb) {
  start(Mode::OpenFile, title, dir, "", extension, std::move(cb));
}

void getSaveFileName(const std::string &title, const std::string &default_path, const std::string &extension, Callback cb) {
  const fs::path p(default_path);
  start(Mode::SaveFile, title, p.parent_path(), p.filename().string(), extension, std::move(cb));
}

void getExistingDirectory(const std::string &title, const std::string &dir, Callback cb) {
  start(Mode::Directory, title, dir, "", "", std::move(cb));
}

bool isOpen() { return g_state.active; }

void draw() {
  State &s = g_state;
  if (!s.active) return;
  const std::string popup_id = s.title + "###FileDialog";
  if (!s.open) {
    ImGui::OpenPopup(popup_id.c_str());
    s.open = true;
  }
  ImGui::SetNextWindowSize(ImVec2(640.0f, 480.0f), ImGuiCond_Appearing);
  ImGui::SetNextWindowPos(ImGui::GetMainViewport()->GetCenter(), ImGuiCond_Appearing, ImVec2(0.5f, 0.5f));
  if (!ImGui::BeginPopupModal(popup_id.c_str(), nullptr, ImGuiWindowFlags_NoSavedSettings)) return;

  if (ImGui::Button("Up")) setDir(s.dir.parent_path());
  ImGui::SameLine();
  ImGui::SetNextItemWidth(-1.0f);
  if (inputText("##dir", &s.dir_input, "", ImGuiInputTextFlags_EnterReturnsTrue)) setDir(s.dir_input);

  const float footer = ImGui::GetFrameHeightWithSpacing() * (s.mode == Mode::Directory ? 1.0f : 2.0f) + ImGui::GetStyle().ItemSpacing.y;
  ImGui::BeginChild("entries", ImVec2(0, -footer), ImGuiChildFlags_Borders);
  std::error_code dir_ec;
  for (size_t i = 0; i < s.entries.size(); ++i) {
    const auto &entry = s.entries[i];
    const bool is_dir = entry.is_directory(dir_ec);
    const std::string name = entry.path().filename().string();
    const std::string label = (is_dir ? std::string(u8"  ") : std::string(u8"  ")) + name;
    ImGui::PushID(static_cast<int>(i));
    const bool selected = !is_dir && name == s.filename;
    if (ImGui::Selectable(label.c_str(), selected, ImGuiSelectableFlags_AllowDoubleClick)) {
      if (is_dir) {
        if (ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left)) {
          ImGui::PopID();
          setDir(entry.path());
          break;
        }
        if (s.mode == Mode::Directory) s.filename = name;
      } else {
        s.filename = name;
        if (ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left) && s.mode == Mode::OpenFile) {
          ImGui::PopID();
          ImGui::CloseCurrentPopup();
          ImGui::EndChild();
          ImGui::EndPopup();
          accept(entry.path());
          return;
        }
      }
    }
    ImGui::PopID();
  }
  ImGui::EndChild();

  bool ok = false, cancel = false;
  if (s.mode != Mode::Directory) {
    ImGui::SetNextItemWidth(-90.0f);
    if (inputText("##name", &s.filename, "File name", ImGuiInputTextFlags_EnterReturnsTrue)) ok = true;
    ImGui::SameLine();
    ImGui::TextDisabled("%s", s.extension.empty() ? "*" : ("*" + s.extension).c_str());
  }
  const char *accept_label = s.mode == Mode::SaveFile ? "Save" : (s.mode == Mode::Directory ? "Choose" : "Open");
  if (ImGui::Button(accept_label, ImVec2(80.0f, 0.0f))) ok = true;
  ImGui::SameLine();
  if (ImGui::Button("Cancel", ImVec2(80.0f, 0.0f)) || ImGui::IsKeyPressed(ImGuiKey_Escape, false)) cancel = true;

  fs::path result;
  if (ok) {
    if (s.mode == Mode::Directory) {
      result = s.filename.empty() ? s.dir : s.dir / s.filename;
    } else if (!s.filename.empty()) {
      result = fs::path(s.filename).is_absolute() ? fs::path(s.filename) : s.dir / s.filename;
      if (s.mode == Mode::SaveFile && !s.extension.empty() && result.extension().empty()) result += s.extension;
      if (s.mode == Mode::OpenFile && !fs::is_regular_file(result, dir_ec)) ok = false;
    } else {
      ok = false;
    }
  }
  if (ok || cancel) ImGui::CloseCurrentPopup();
  ImGui::EndPopup();
  if (cancel) {
    finish("");
  } else if (ok) {
    accept(result);
  }
}

}  // namespace FileDialog
