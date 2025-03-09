#pragma once

#include <QWidget>
#include <QVBoxLayout>
#include <QProgressBar>
#include <QLabel>
#include "common/params.h"

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
  
  ParamControl *enable_firehose;
  QFrame *progress_container;
  QProgressBar *progress_bar;
  QLabel *progress_text;
  QLabel *detailed_instructions;
  
  void updateFirehoseState(bool enabled);
};
