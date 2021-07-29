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
  explicit Updater(QWidget *parent = 0);

private slots:
  void installUpdate();
  void readProgress();
  void updateFinished(int exitCode, QProcess::ExitStatus exitStatus);

private:
  QLabel *text;
  QProgressBar *bar;
  QPushButton *reboot;
	QProcess proc;

  QWidget *prompt, *wifi, *progress;
};
