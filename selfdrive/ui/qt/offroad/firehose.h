#pragma once

#include <QWidget>
#include <QVBoxLayout>
#include <QProgressBar>
#include <QLabel>
#include "selfdrive/ui/qt/widgets/controls.h"
#include "common/params.h"

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
