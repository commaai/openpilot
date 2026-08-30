#pragma once

// non-modal dialogs drawn by MainWindow every frame until closed
class ToolDialog {
public:
  virtual ~ToolDialog() = default;
  virtual bool draw() = 0;  // false once the dialog was closed
};
