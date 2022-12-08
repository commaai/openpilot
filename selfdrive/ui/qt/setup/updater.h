#pragma once

#include <QLabel>
#include <QProcess>
#include <QPushButton>
#include <QProgressBar>
#include <QStackedWidget>
#include <QWidget>

class Updater : public QStackedWidget {
  Q_OBJECT

public:
  explicit Updater(const QString &updater_path, const QString &manifest_path, QWidget *parent = 0);

private slots:
  void installUpdate();
  void readProgress();
  void updateFinished(int exitCode, QProcess::ExitStatus exitStatus);

private:
  QProcess proc;
  QString updater, manifest;

  QLabel *text;
  QProgressBar *bar;
  QPushButton *reboot;
  QWidget *prompt, *wifi, *progress;
};
