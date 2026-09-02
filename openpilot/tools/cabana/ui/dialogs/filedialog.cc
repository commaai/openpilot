#include "tools/cabana/ui/dialogs/filedialog.h"

#include <algorithm>
#include <cctype>
#include <system_error>

#include "imgui.h"
#include "imgui_internal.h"
#include "tools/cabana/ui/dialogs/messagebox.h"
#include "tools/cabana/ui/icons.h"
#include "tools/cabana/ui/util.h"

namespace fs = std::filesystem;

namespace FileDialog {

namespace {

// case insensitive, numeric aware collation ("mazda_3_2019" before "mazda_2017"): punctuation is ignored at
// the first level ("FORD_CADS_64" before "FORD_CADS.dbc"), digit runs compare numerically, case is ignored;
// ties fall back to a plain comparison
bool naturalLess(const std::string &a, const std::string &b) {
  auto skip = [](const std::string &s, size_t &i) {
    while (i < s.size() && !isalnum(static_cast<unsigned char>(s[i]))) ++i;
  };
  size_t i = 0, j = 0;
  for (;;) {
    skip(a, i);
    skip(b, j);
    if (i >= a.size() || j >= b.size()) break;
    if (isdigit(static_cast<unsigned char>(a[i])) && isdigit(static_cast<unsigned char>(b[j]))) {
      size_t ie = i, je = j;
      while (ie < a.size() && isdigit(static_cast<unsigned char>(a[ie]))) ++ie;
      while (je < b.size() && isdigit(static_cast<unsigned char>(b[je]))) ++je;
      const unsigned long long na = std::stoull(a.substr(i, ie - i)), nb = std::stoull(b.substr(j, je - j));
      if (na != nb) return na < nb;
      i = ie;
      j = je;
    } else {
      const int ca = tolower(static_cast<unsigned char>(a[i])), cb = tolower(static_cast<unsigned char>(b[j]));
      if (ca != cb) return ca < cb;
      ++i;
      ++j;
    }
  }
  const bool a_done = i >= a.size(), b_done = j >= b.size();
  if (a_done != b_done) return a_done;
  return a < b;
}

enum class Mode { OpenFile, SaveFile, Directory };

struct State {
  bool active = false;
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
PopupOwner g_owner;

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
    return da != db ? da : naturalLess(a.path().filename().string(), b.path().filename().string());
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
  g_owner.reset();
  setDir(dir);
}

void finish(const std::string &path) {
  Callback cb = std::move(g_state.callback);
  g_state = State{};
  g_owner.reset();
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

void draw() {
  State &s = g_state;
  if (!s.active) return;
  const std::string popup_id = s.title + "###FileDialog";
  if (!beginDialog(popup_id.c_str(), &g_owner, ImVec2(640.0f, 480.0f), 0)) return;

  if (ImGui::Button("Up")) setDir(s.dir.parent_path());
  ImGui::SameLine();
  ImGui::SetNextItemWidth(-1.0f);
  if (inputText("##dir", &s.dir_input, "", ImGuiInputTextFlags_EnterReturnsTrue)) setDir(s.dir_input);

  const float footer = ImGui::GetFrameHeightWithSpacing() * (s.mode == Mode::Directory ? 1.0f : 2.0f) + ImGui::GetStyle().ItemSpacing.y;
  bool ok = false, cancel = false;
  fs::path result, pending_dir;
  ImGui::BeginChild("entries", ImVec2(0, -footer), ImGuiChildFlags_Borders);
  std::error_code dir_ec;
  for (size_t i = 0; i < s.entries.size(); ++i) {
    const auto &entry = s.entries[i];
    const bool is_dir = entry.is_directory(dir_ec);
    const std::string name = entry.path().filename().string();
    const std::string label = (is_dir ? std::string(icon::FOLDER) : std::string(icon::FILE_EARMARK)) + "  " + name;
    ImGui::PushID(static_cast<int>(i));
    const bool selected = !is_dir && name == s.filename;
    if (ImGui::Selectable(label.c_str(), selected, ImGuiSelectableFlags_AllowDoubleClick)) {
      const bool double_clicked = ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left);
      if (is_dir) {
        if (double_clicked) {
          pending_dir = entry.path();
        } else if (s.mode == Mode::Directory) {
          s.filename = name;
        }
      } else {
        s.filename = name;
        if (double_clicked && s.mode == Mode::OpenFile) {
          result = entry.path();
          ok = true;
        }
      }
    }
    ImGui::PopID();
    if (ok || !pending_dir.empty()) break;
  }
  ImGui::EndChild();
  if (!pending_dir.empty()) setDir(pending_dir);

  if (s.mode != Mode::Directory) {
    ImGui::SetNextItemWidth(-90.0f);
    if (inputText("##name", &s.filename, "File name", ImGuiInputTextFlags_EnterReturnsTrue)) ok = true;
    ImGui::SameLine();
    ImGui::TextDisabled("%s", s.extension.empty() ? "*" : ("*" + s.extension).c_str());
  }
  const char *accept_label = s.mode == Mode::SaveFile ? "Save" : (s.mode == Mode::Directory ? "Choose" : "Open");
  dialogButtons(accept_label, &ok, &cancel);

  if (ok && result.empty()) {
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
  // nested so the overwrite prompt stacks on this dialog; it may finish the dialog through accept()
  MessageBox::draw();
  if (!s.active) ImGui::CloseCurrentPopup();
  ImGui::EndPopup();
  if (cancel) {
    finish("");
  } else if (ok) {
    accept(result);
  }
}

}  // namespace FileDialog
