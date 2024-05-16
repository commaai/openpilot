#pragma once

#include <QWidget>

#include "selfdrive/ui/ui.h"

class OnroadAlerts : public QWidget {
  Q_OBJECT

public:
  OnroadAlerts(QWidget *parent = 0) : QWidget(parent) {}
  void updateState(const UIState &s);
  void clear();

protected:
  struct Alert {
    QString text1;
    QString text2;
    QString type;
    cereal::ControlsState::AlertSize size;
    cereal::ControlsState::AlertStatus status;

    bool equal(const Alert &other) const {
      return text1 == other.text1 && text2 == other.text2 && type == other.type;
    }
  };

  const QMap<cereal::ControlsState::AlertStatus, QColor> alert_colors = {
    {cereal::ControlsState::AlertStatus::NORMAL, QColor(0x15, 0x15, 0x15, 0xf1)},
    {cereal::ControlsState::AlertStatus::USER_PROMPT, QColor(0xDA, 0x6F, 0x25, 0xf1)},
    {cereal::ControlsState::AlertStatus::CRITICAL, QColor(0xC9, 0x22, 0x31, 0xf1)},
  };

  void paintEvent(QPaintEvent*) override;
  OnroadAlerts::Alert getAlert(const SubMaster &sm, uint64_t started_frame);

  QColor bg;
  Alert alert = {};
};
