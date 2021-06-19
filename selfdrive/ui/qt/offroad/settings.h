#pragma once

#include <QButtonGroup>
#include <QFileSystemWatcher>
#include <QFrame>
#include <QGraphicsView>
#include <QLabel>
#include <QPushButton>
#include <QStackedWidget>
#include <QWidget>


#include "selfdrive/ui/qt/widgets/controls.h"

// ********** settings window + top-level panels **********

class DevicePanel : public QWidget {
  Q_OBJECT
public:
  explicit DevicePanel(QWidget* parent = nullptr);
signals:
  void reviewTrainingGuide();
  void showDriverView();
};

class TogglesPanel : public QWidget {
  Q_OBJECT
public:
  explicit TogglesPanel(QWidget *parent = nullptr);
};

#include <QPainter>
#include <QPaintEngine>
#include <QDebug>
class SoftwarePanel : public QWidget {
  Q_OBJECT
public:
  explicit SoftwarePanel(QWidget* parent = nullptr);
  void paintEvent(QPaintEvent *event) override{
    QPainter p(this);
    qInfo() << p.paintEngine()->type();
    assert(p.paintEngine()->type() == QPaintEngine::OpenGL2);
  }
private:
  void showEvent(QShowEvent *event) override;
  void updateLabels();

  LabelControl *gitBranchLbl;
  LabelControl *gitCommitLbl;
  LabelControl *osVersionLbl;
  LabelControl *versionLbl;
  LabelControl *lastUpdateLbl;
  ButtonControl *updateBtn;

  Params params;
  QFileSystemWatcher *fs_watch;
};

class QGraphicsProxyWidget;

class SettingsWindow : public QGraphicsView {
  Q_OBJECT

public:
  explicit SettingsWindow(QWidget *parent = 0);

protected:
  void hideEvent(QHideEvent *event) override;
  void showEvent(QShowEvent *event) override;
  void resizeEvent(QResizeEvent *event) override;

signals:
  void closeSettings();
  void offroadTransition(bool offroad);
  void reviewTrainingGuide();
  void showDriverView();

private:
  QPushButton *sidebar_alert_widget;
  QWidget *sidebar_widget;
  QButtonGroup *nav_btns;
  QStackedWidget *panel_widget;
  QGraphicsProxyWidget *proxyWidget;
};
