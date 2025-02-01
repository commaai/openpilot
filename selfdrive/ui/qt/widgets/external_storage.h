#pragma once

#include <QPushButton>

#include "system/hardware/hw.h"
#include "selfdrive/ui/qt/widgets/controls.h"

class ExternalStorageControl : public ButtonControl {
  Q_OBJECT

public:
  ExternalStorageControl();

protected:
  void showEvent(QShowEvent *event) override;

private:
  Params params;

  void refresh();
  bool isStorageMounted();
  bool isFilesystemPresent();
  bool isDriveInitialized();
  void mountStorage();
  void unmountStorage();
  void formatStorage();
  void checkAndUpdateFstab();
};
