#pragma once

#include <QTimer>
#include <QWidget>
#include <QNetworkAccessManager>

class SSH : public QWidget {
  Q_OBJECT

public:
  explicit SSH(QWidget* parent = 0);

private:
  QNetworkAccessManager* manager;
  QNetworkReply* reply;
  QTimer* networkTimer;
  bool aborted;

  void getSSHKeys(QString user);

signals:
  void NoSSHAdded();
  void SSHAdded();
  void failedResponse(QString errorString);
  void gotSSHKeys();
  void closeSSHSettings();

private slots:
  void checkForSSHKey();
  void timeout();
  void parseResponse();
};
