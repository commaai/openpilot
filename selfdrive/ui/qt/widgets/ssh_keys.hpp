#pragma once

#include <QTimer>
#include <QPushButton>
#include <QNetworkAccessManager>

#include "widgets/controls.hpp"

class SshControl : public AbstractControl {
  Q_OBJECT

public:
  SshControl();

private:
  QPushButton btn;
  QString username;

  // networking
  QTimer* networkTimer;
  QNetworkReply* reply;
  QNetworkAccessManager* manager;

  void refresh();
  void getUserKeys(QString username);

signals:
  void failedResponse(QString errorString);

private slots:
  void timeout();
  void parseResponse();
};
