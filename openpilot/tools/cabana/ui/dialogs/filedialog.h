#pragma once

#include <filesystem>
#include <functional>
#include <string>
#include <vector>

// file browser. One dialog at a time; the callback gets an empty path on cancel.
namespace FileDialog {

using Callback = std::function<void(const std::string &path)>;

void getOpenFileName(const std::string &title, const std::string &dir, const std::string &extension, Callback cb);
void getSaveFileName(const std::string &title, const std::string &default_path, const std::string &extension, Callback cb);
void getExistingDirectory(const std::string &title, const std::string &dir, Callback cb);

void draw();  // once per frame, at the top popup level

}  // namespace FileDialog
