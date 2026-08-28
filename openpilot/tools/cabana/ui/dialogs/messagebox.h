#pragma once

#include <functional>
#include <string>

// QMessageBox equivalent. Boxes are queued and shown one at a time as modal popups (Qt's exec() blocks the
// caller; here the caller passes a continuation where it needs the answer).
namespace MessageBox {

void information(const std::string &title, const std::string &text);
void warning(const std::string &title, const std::string &text, const std::string &detailed_text = "");
// Ok | Cancel; on_result(true) for Ok
void question(const std::string &title, const std::string &text, std::function<void(bool ok)> on_result);

bool isOpen();
void draw();  // once per frame, at the top popup level

}  // namespace MessageBox
