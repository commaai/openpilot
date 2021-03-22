#pragma once

#include <QWidget>
#include <QButtonGroup>
#include <QVBoxLayout>
#include <QStackedWidget>
#include <QPushButton>
#include <QTimer>
#include <QNetworkAccessManager>
#include <QStackedLayout>

#include "widgets/input.hpp"
#include "widgets/controls.hpp"

class SshControl : public AbstractControl {
  Q_OBJECT

public:
  SshControl();

private:
  QPushButton btn;
  QNetworkAccessManager* manager;

  QString usernameGitHub;
  QNetworkReply* reply;
  QTimer* networkTimer;
  bool aborted;

signals:
  void NoSSHAdded();
  void SSHAdded();
  void failedResponse(QString errorString);
  void gotSSHKeys();

private slots:
  void checkForSSHKey();
  void getSSHKeys(QString username);
  void timeout();
  void parseResponse();
};
