#pragma once

// non-modal Qt dialogs (show()) drawn by MainWindow every frame until closed
class ToolDialog {
public:
  virtual ~ToolDialog() = default;
  virtual bool draw() = 0;  // false once the dialog was closed
};
