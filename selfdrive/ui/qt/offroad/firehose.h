#pragma once

#include <QWidget>
#include <QVBoxLayout>
#include <QProgressBar>
#include <QLabel>
#include "selfdrive/ui/qt/widgets/controls.h"
#include "selfdrive/ui/qt/request_repeater.h"
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
  QLabel *detailed_instructions;
  QLabel *contribution_label;
  QLabel *toggle_label;

  RequestRepeater *firehose_stats;

private slots:
  void refresh();
};
