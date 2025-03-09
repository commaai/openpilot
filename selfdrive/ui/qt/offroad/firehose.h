#pragma once

#include <QWidget>
#include <QVBoxLayout>
#include <QLabel>
#include "selfdrive/ui/qt/request_repeater.h"

#ifdef SUNNYPILOT
#include "selfdrive/ui/sunnypilot/ui.h"
#include "selfdrive/ui/sunnypilot/qt/widgets/controls.h"
#include "selfdrive/ui/sunnypilot/qt/offroad/settings/settings.h"
#else
#include "selfdrive/ui/ui.h"
#include "selfdrive/ui/qt/widgets/controls.h"
#include "selfdrive/ui/qt/offroad/settings.h"
#endif

// Forward declarations
class SettingsWindow;

class FirehosePanel : public QWidget {
  Q_OBJECT
public:
  explicit FirehosePanel(SettingsWindow *parent);

private:
  QVBoxLayout *layout;

  QLabel *detailed_instructions;
  QLabel *contribution_label;
  QLabel *toggle_label;

  RequestRepeater *firehose_stats;

private slots:
  void refresh();
};
