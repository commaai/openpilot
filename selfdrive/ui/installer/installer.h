#pragma once

#include <QLabel>
#include <QProcess>
#include <QProgressBar>
#include <QWidget>

class Installer : public QWidget {
  Q_OBJECT

public:
  explicit Installer(QWidget *parent = 0);

private slots:
  void updateProgress(int percent);

  void readProgress();
  void cloneFinished(int exitCode, QProcess::ExitStatus exitStatus);

private:
  QLabel *val;
  QProgressBar *bar;
  QProcess proc;

  void doInstall();
  void freshClone();
  void cachedFetch(const QString &cache);
};
