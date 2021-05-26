#pragma once

#include <QButtonGroup>
#include <QFrame>
#include <QLabel>
#include <QPushButton>
#include <QScrollArea>
#include <QStackedWidget>
#include <QTimer>
#include <QWidget>

#include "selfdrive/ui/qt/widgets/scrollview.h"
#include "selfdrive/ui/qt/widgets/controls.h"

// ********** settings window + top-level panels **********

class DevicePanel : public QWidget {
  Q_OBJECT
public:
  explicit DevicePanel(QWidget* parent = nullptr);
signals:
  void reviewTrainingGuide();
};

class TogglesPanel : public QWidget {
  Q_OBJECT
public:
  explicit TogglesPanel(QWidget *parent = nullptr);
};

class ReleaseNotes : public QFrame {
  Q_OBJECT

public:
  explicit ReleaseNotes(QWidget *parent = 0);

private:
  Params params;
  QLabel releaseNotes;
  ScrollView *releaseNotesScroll;
};

class UpdateResult : public QFrame {
  Q_OBJECT

public:
  explicit UpdateResult(QWidget *parent = 0);
  void setText(const QString &text) { result.setText(text); }
  void setBtnVisible(bool visible) { rebootBtn.setVisible(visible); }

private:
  Params params;
  ScrollView *resultScroll;
  QLabel result;
  QPushButton rebootBtn;
};

class UpdatePanel : public QWidget {
  Q_OBJECT
public:
  explicit UpdatePanel(QWidget *parent = nullptr);
  void closeReleaseNote();
  void closeDownloadUpdate();

private slots:
  void refreshUpdate();

private:
  ReleaseNotes* releaseNotes;
  UpdateResult* updateResult;
  ButtonControl* viewReleaseNoteBtn; 
  ButtonControl* downloadUpdateBtn;
  Params params;
  std::string updateFailedCount;
  std::string lastUpdate;
  QTimer *timer;
  bool isDownloading = false;
};

class DeveloperPanel : public QFrame {
  Q_OBJECT
public:
  explicit DeveloperPanel(QWidget* parent = nullptr);

protected:
  void showEvent(QShowEvent *event) override;
  QList<LabelControl *> labels;
};

class SettingsWindow : public QFrame {
  Q_OBJECT

public:
  explicit SettingsWindow(QWidget *parent = 0) : QFrame(parent) {};

protected:
  void hideEvent(QHideEvent *event);
  void showEvent(QShowEvent *event);

signals:
  void closeSettings();
  void offroadTransition(bool offroad);
  void reviewTrainingGuide();

private:
  QPushButton *sidebar_alert_widget;
  QWidget *sidebar_widget;
  QButtonGroup *nav_btns;
  QStackedWidget *panel_widget;
};
