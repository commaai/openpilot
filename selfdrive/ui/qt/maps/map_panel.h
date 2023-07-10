#pragma once

#include <QFrame>
#include <QMapboxGL>
#include <QStackedLayout>

class MapPanel : public QFrame {
  Q_OBJECT

public:
  explicit MapPanel(const QMapboxGLSettings &settings, QWidget *parent = nullptr);

  bool isShowingMap() const;

signals:
  void mapPanelRequested();

private:
  QStackedLayout *content_stack;
};
