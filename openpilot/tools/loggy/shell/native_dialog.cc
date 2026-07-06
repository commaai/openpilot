#include "tools/loggy/shell/native_dialog.h"

#include <array>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <optional>
#include <string_view>

namespace loggy {

std::string shell_quote(std::string_view value) {
  std::string quoted;
  quoted.reserve(value.size() + 2);
  quoted.push_back('\'');
  for (char c : value) {
    if (c == '\'') quoted += "'\\''";
    else quoted.push_back(c);
  }
  quoted.push_back('\'');
  return quoted;
}

namespace {

void append_arg(std::string &command, std::string_view arg) {
  if (!command.empty()) command.push_back(' ');
  command += shell_quote(arg);
}

bool executable_exists_in_path(std::string_view name) {
  const char *path_env = std::getenv("PATH");
  if (name.empty() || path_env == nullptr) return false;
  std::string_view paths(path_env);
  for (size_t start_ = 0; start_ <= paths.size();) {
    const size_t end = paths.find(':', start_);
    std::filesystem::path dir(paths.substr(start_, end == std::string_view::npos ? paths.size() - start_ : end - start_));
    if (dir.empty()) dir = ".";
    std::error_code ec;
    if (std::filesystem::exists(dir / std::string(name), ec) && !ec) return true;
    if (end == std::string_view::npos) break;
    start_ = end + 1;
  }
  return false;
}

std::optional<std::string_view> native_dialog_backend() {
#ifdef __APPLE__
  if (executable_exists_in_path("osascript")) return std::string_view("osascript");
#endif
  if (executable_exists_in_path("zenity")) return std::string_view("zenity");
  if (executable_exists_in_path("kdialog")) return std::string_view("kdialog");
  if (executable_exists_in_path("osascript")) return std::string_view("osascript");
  return std::nullopt;
}

std::string osascript_escape(std::string_view value) {
  std::string out;
  out.reserve(value.size());
  for (char c : value) {
    if (c == '\\' || c == '"') out.push_back('\\');
    out.push_back(c);
  }
  return out;
}

void append_title(std::string &command, std::string_view title, std::string_view flag = "--title") {
  if (title.empty()) return;
  append_arg(command, flag);
  append_arg(command, title);
}

std::string native_dialog_command(NativeDialogType type, const NativeDialogOptions &options, std::string_view backend) {
  std::string command;
  if (backend == "zenity") {
    append_arg(command, "zenity");
    append_arg(command, "--file-selection");
    if (type == NativeDialogType::SaveFile) append_arg(command, "--save");
    if (type == NativeDialogType::SelectDirectory) append_arg(command, "--directory");
    if (options.confirm_overwrite) append_arg(command, "--confirm-overwrite");
    append_title(command, options.title);
    if (!options.path.empty()) {
      append_arg(command, "--filename");
      append_arg(command, options.path);
    }
  } else if (backend == "kdialog") {
    append_arg(command, "kdialog");
    append_arg(command, type == NativeDialogType::OpenFile ? "--getopenfilename"
                        : type == NativeDialogType::SaveFile ? "--getsavefilename"
                                                              : "--getexistingdirectory");
    if (options.confirm_overwrite && type == NativeDialogType::SaveFile) append_arg(command, "--overwrite");
    append_title(command, options.title);
    if (!options.path.empty()) {
      append_arg(command, "--default");
      append_arg(command, options.path);
    }
  } else {
    append_arg(command, "osascript");
    append_arg(command, "-e");
    std::string script = "choose ";
    script += type == NativeDialogType::OpenFile ? "file"
            : type == NativeDialogType::SaveFile ? "file name"
                                                 : "folder";
    if (!options.title.empty()) script += " with prompt \"" + osascript_escape(options.title) + "\"";
    append_arg(command, script);
  }
  command += " 2>/dev/null";
  return command;
}

std::string first_output_line(std::string_view output) {
  size_t begin = 0;
  while (begin < output.size() && std::isspace(static_cast<unsigned char>(output[begin])) != 0) ++begin;
  size_t end = output.find_first_of("\r\n", begin);
  if (end == std::string_view::npos) end = output.size();
  while (end > begin && std::isspace(static_cast<unsigned char>(output[end - 1])) != 0) --end;
  return std::string(output.substr(begin, end - begin));
}

}  // namespace

std::optional<std::string> native_dialog_choose_path(NativeDialogType type,
                                                     const NativeDialogOptions &options,
                                                     std::string &error) {
  const std::optional<std::string_view> backend = native_dialog_backend();
  if (!backend.has_value()) {
    error = "No native file dialog backend found (expected zenity, kdialog, or osascript)";
    return std::nullopt;
  }

  FILE *pipe = popen(native_dialog_command(type, options, *backend).c_str(), "r");
  if (pipe == nullptr) {
    error = "Failed to launch " + std::string(*backend);
    return std::nullopt;
  }

  std::string output;
  std::array<char, 512> buffer{};
  while (fgets(buffer.data(), static_cast<int>(buffer.size()), pipe) != nullptr) {
    output += buffer.data();
  }
  pclose(pipe);
  const std::string path = first_output_line(output);
  if (!path.empty()) {
    error.clear();
    return path;
  }
  error.clear();
  return std::nullopt;
}

}  // namespace loggy
