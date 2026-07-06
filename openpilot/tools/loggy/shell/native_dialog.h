#pragma once

#include <optional>
#include <string>
#include <string_view>

namespace loggy {

// Single-quotes value for safe inclusion in a shell command line, escaping embedded quotes.
std::string shell_quote(std::string_view value);

enum class NativeDialogType {
  OpenFile,
  SaveFile,
  SelectDirectory,
};

struct NativeDialogOptions {
  std::string title;
  std::string path;
  bool confirm_overwrite = false;
};

std::optional<std::string> native_dialog_choose_path(NativeDialogType type,
                                                     const NativeDialogOptions &options,
                                                     std::string &error);

}  // namespace loggy
