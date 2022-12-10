#pragma once

#include <QSet>

#include "selfdrive/ui/qt/widgets/controls.h"

class ReplayPanel : public ListWidget {
  Q_OBJECT

public:
  ReplayPanel(QWidget *parent);

public slots:
  void replayRoute(const QString &route);
  void stopReplay();

protected:
  void showEvent(QShowEvent *event) override;

  QSet<QString> route_names;
};
