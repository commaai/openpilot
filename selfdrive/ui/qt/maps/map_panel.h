#pragma once

#include <QFrame>
#include <QMapLibre/Settings>
#include <QStackedLayout>

class MapPanel : public QFrame {
  Q_OBJECT

public:
  explicit MapPanel(const QMapLibre::Settings &settings, QWidget *parent = nullptr);

signals:
  void mapPanelRequested();

public slots:
  void toggleMapSettings();

private:
  QStackedLayout *content_stack;
};
