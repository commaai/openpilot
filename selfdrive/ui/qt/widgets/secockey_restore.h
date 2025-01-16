#pragma once

#include <QPushButton>
#include <QFile>

#include "selfdrive/ui/qt/widgets/controls.h"

class SecOCKeyRestore : public ButtonControl {
  Q_OBJECT

public:
  SecOCKeyRestore();

private:
  Params params;

  QString getArchive(QString);
  bool isValid(QString);
  void refresh();
};
