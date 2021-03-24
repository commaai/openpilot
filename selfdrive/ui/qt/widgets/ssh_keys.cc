#include <QNetworkReply>

#include "widgets/input.hpp"
#include "widgets/ssh_keys.hpp"
#include "common/params.h"


SshControl::SshControl() : AbstractControl("SSH Keys", "Warning: This grants SSH access to all public keys in your GitHub settings. Never enter a GitHub username other than your own. A comma employee will NEVER ask you to add their GitHub username.", "") {
  // setup widget
  btn.setStyleSheet(R"(
    padding: 0;
    border-radius: 50px;
    font-size: 40px;
    font-weight: 500;
    color: #E4E4E4;
    background-color: #393939;
  )");
  btn.setFixedSize(250, 100);
  hlayout->addWidget(&btn);

  QObject::connect(&btn, &QPushButton::released, [=]() {
    if (btn.text() == "ADD") {
      username = InputDialog::getText("Enter your GitHub username");
      if (username.length() > 0) {
        btn.setText("LOADING");
        btn.setEnabled(false);
        getUserKeys(username);
      }
    } else {
      Params().delete_db_value("GithubSshKeys");
      refresh();
    }
  });

  // setup networking
  manager = new QNetworkAccessManager(this);
  networkTimer = new QTimer(this);
  networkTimer->setSingleShot(true);
  networkTimer->setInterval(5000);
  connect(networkTimer, SIGNAL(timeout()), this, SLOT(timeout()));

  refresh();
}

void SshControl::refresh() {
  QString param = QString::fromStdString(Params().get("GithubSshKeys"));
  if (param.length()) {
    btn.setText("REMOVE");
  } else {
    btn.setText("ADD");
  }
  btn.setEnabled(true);
}

void SshControl::getUserKeys(QString username){
  QString url = "https://github.com/" + username + ".keys";

  QNetworkRequest request;
  request.setUrl(QUrl(url));
#ifdef QCOM
  QSslConfiguration ssl = QSslConfiguration::defaultConfiguration();
  ssl.setCaCertificates(QSslCertificate::fromPath("/usr/etc/tls/cert.pem",
                        QSsl::Pem, QRegExp::Wildcard));
  request.setSslConfiguration(ssl);
#endif

  reply = manager->get(request);
  connect(reply, SIGNAL(finished()), this, SLOT(parseResponse()));
  networkTimer->start();
}

void SshControl::timeout(){
  reply->abort();
}

void SshControl::parseResponse(){
  QString err = "";
  if (reply->error() != QNetworkReply::OperationCanceledError) {
    networkTimer->stop();
    QString response = reply->readAll();
    if (reply->error() == QNetworkReply::NoError && response.length()) {
      Params().write_db_value("GithubSshKeys", response.toStdString());
    } else if(reply->error() == QNetworkReply::NoError){
      err = "Username '" + username + "' has no keys on GitHub";
    } else {
      err = "Username '" + username + "' doesn't exist on GitHub";
    }
  } else {
    err = "Request timed out";
  }

  if (err.length()) {
    ConfirmationDialog::alert(err);
  }

  refresh();
  reply->deleteLater();
  reply = nullptr;
}
