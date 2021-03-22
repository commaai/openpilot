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
  QString username;

  // networking
  bool aborted;
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
