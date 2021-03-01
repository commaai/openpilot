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

class SSH : public QWidget {
  Q_OBJECT

public:
  explicit SSH(QWidget* parent = 0);

private:
  QStackedLayout* slayout;
  InputDialog* dialog;
  QNetworkAccessManager* manager;

  QString usernameGitHub;
  QNetworkReply* reply;
  QTimer* networkTimer;
  bool aborted;

signals:
  void closeSSHSettings();
  void NoSSHAdded();
  void SSHAdded();
  void failedResponse(QString errorString);
  void gotSSHKeys();

private slots:
  void checkForSSHKey();
  void getSSHKeys();
  void timeout();
  void parseResponse();
};

