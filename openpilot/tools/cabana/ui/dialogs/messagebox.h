#pragma once

#include <functional>
#include <string>

// Boxes are queued and shown one at a time as modal popups; the caller passes a continuation where it
// needs the answer.
namespace MessageBox {

// on_close (optional) runs when the box is dismissed
void information(const std::string &title, const std::string &text, std::function<void()> on_close = nullptr);
void warning(const std::string &title, const std::string &text, const std::string &detailed_text = "",
             std::function<void()> on_close = nullptr);
// Ok | Cancel; on_result(true) for Ok
void question(const std::string &title, const std::string &text, std::function<void(bool ok)> on_result);

void draw();  // once per frame, at the top popup level

}  // namespace MessageBox
