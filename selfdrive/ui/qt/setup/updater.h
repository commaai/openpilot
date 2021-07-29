#pragma once

#include <QLabel>
#include <QProcess>
#include <QProgressBar>
#include <QStackedWidget>
#include <QWidget>

class Updater : public QStackedWidget {
  Q_OBJECT

public:
  explicit Updater(QWidget *parent = 0);

private slots:
  void updateProgress(int percent);

  void installUpdate();

  void readProgress();
  void updateFinished(int exitCode, QProcess::ExitStatus exitStatus);

private:
  QLabel *val;
  QProgressBar *bar;
  QProcess proc;

  // three views
  QWidget *prompt, *wifi, *progress;
};
