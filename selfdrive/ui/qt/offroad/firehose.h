#pragma once

#include <QWidget>
#include <QVBoxLayout>
#include <QLabel>
#include "selfdrive/ui/qt/request_repeater.h"

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
