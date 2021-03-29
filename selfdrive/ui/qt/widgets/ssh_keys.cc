#include <QNetworkReply>
#include <QHBoxLayout>
#include "widgets/input.hpp"
#include "widgets/ssh_keys.hpp"
#include "common/params.h"


SshControl::SshControl() : AbstractControl("SSH Keys", "Warning: This grants SSH access to all public keys in your GitHub settings. Never enter a GitHub username other than your own. A comma employee will NEVER ask you to add their GitHub username.", "") {

  // setup widget
  hlayout->addStretch(1);

  username_label.setAlignment(Qt::AlignVCenter);
  username_label.setStyleSheet("color: #aaaaaa");
  hlayout->addWidget(&username_label);

  btn.setStyleSheet(R"(
    padding: 0;
    border-radius: 50px;
    font-size: 35px;
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
      Params().remove("GithubUsername");
      Params().remove("GithubSshKeys");
      refresh();
    }
  });

  refresh();
}

void SshControl::refresh() {
  QString param = QString::fromStdString(Params().get("GithubSshKeys"));
  if (param.length()) {
    username_label.setText(QString::fromStdString(Params().get("GithubUsername")));
    btn.setText("REMOVE");
  } else {
    username_label.setText("");
    btn.setText("ADD");
  }
  btn.setEnabled(true);
}

void SshControl::getUserKeys(QString username){
  const QString url = "https://github.com/" + username + ".keys";
  auto [err, resp] = httpGet(url, 20000);
  if (err == QNetworkReply::NoError) {
    if (!resp.isEmpty()) {
      Params().write_db_value("GithubUsername", username.toStdString());
      Params().write_db_value("GithubSshKeys", resp.toStdString());
    } else {
      ConfirmationDialog::alert("Username '" + username + "' has no keys on GitHub");
    }
  } else if (err == QNetworkReply::TimeoutError) {
    ConfirmationDialog::alert("Request timed out");
  } else {
    ConfirmationDialog::alert("Username '" + username + "' doesn't exist on GitHub");
  }
  refresh();
}
