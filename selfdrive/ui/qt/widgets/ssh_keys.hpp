#pragma once

#include <QWidget>
#include <QButtonGroup>
#include <QVBoxLayout>
#include <QStackedWidget>
#include <QPushButton>
#include <QTimer>
#include <QNetworkAccessManager>

#include "widgets/input_field.hpp"

class SSH : public QWidget {
  Q_OBJECT

public:
  explicit SSH(QWidget* parent = 0);

private:
  InputField* inputField;
  QStackedLayout* slayout;
  QString usernameGitHub;
  QNetworkAccessManager* manager;
  QNetworkReply* reply;
  QTimer* networkTimer;
  bool aborted;

signals:
  void closeSSHSettings();
  void openKeyboard();
  void closeKeyboard();
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

