#pragma once

#include <functional>
#include <string>

// Minimal, dependency-free ImGui file browser modal used by dbc_menus.cc for
// DBC open/save flows (parity target: QFileDialog::getOpenFileName /
// getSaveFileName as used by tools/cabana/mainwin.cc). Only one instance can
// be open at a time; calling file_dialog_open() while another request is
// still pending replaces it.

enum class FileDialogMode { Open, Save };

// Opens the modal on the next file_dialog_draw() call.
//   mode:              Open requires the resulting path to be an existing
//                       regular file; Save does not.
//   title:              shown inside the modal body (not the OS-level title
//                       bar, to keep the popup's ImGui ID stable across
//                       calls with different titles).
//   start_dir:          initial directory (falls back to cwd if invalid).
//   filter_ext:         e.g. ".dbc"; only files with this suffix are listed
//                       (case-insensitive). Directories are always listed.
//                       Empty = no filter.
//   default_filename:   pre-fills the filename field (Save mode).
//   on_ok:               invoked with the chosen absolute-ish path (start_dir
//                       joined with the current directory navigation) once
//                       the user confirms. Not called on Cancel.
void file_dialog_open(FileDialogMode mode, const std::string &title, const std::string &start_dir,
                       const std::string &filter_ext, const std::string &default_filename,
                       std::function<void(const std::string &path)> on_ok);

// Draws the modal if a request is pending/active. Call once per frame.
void file_dialog_draw();
