#pragma once

#include <optional>
#include <string>

namespace loggy {

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
