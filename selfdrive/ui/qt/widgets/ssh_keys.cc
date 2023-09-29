#include "selfdrive/ui/qt/widgets/ssh_keys.h"

#include "common/params.h"
#include "selfdrive/ui/qt/api.h"
#include "selfdrive/ui/qt/widgets/input.h"

SshControl::SshControl() :
  ButtonControl(tr("SSH Keys"), "", "") {

  QObject::connect(this, &ButtonControl::clicked, [=]() {
    if (text() == tr("ADD")) {
      QString username = InputDialog::getText(tr("Enter GitHub usernames"), this);
      if (username.length() > 0) {
        QStringList usernames = username.split(",");
        setText(tr("LOADING"));
        setEnabled(false);
        getUserKeys(usernames);
      }
    } else {
      clearParams();
      refresh();
    }
  });

  refresh();
}

void SshControl::refresh() {
  QString param = QString::fromStdString(params.get("GithubSshKeys"));
  QString users = QString::fromStdString(params.get("GithubUsername")).split(",").join(", ");
  QString desc = tr("Warning: This grants SSH access to all public keys in the added GitHub accounts' settings. Never enter a "
                    "GitHub username other than your own. A comma employee will NEVER ask you to add their GitHub username.");
  if (param.length()) {
    setValue(QString::fromStdString(params.get("GithubUsername")));
    setText(tr("REMOVE"));
    setDescription(desc + "\n\n" + tr("Added GitHub accounts: %1").arg(users));
  } else {
    setValue("");
    setText(tr("ADD"));
    setDescription(desc);
  }
  setEnabled(true);
}

void SshControl::getUserKeys(const QStringList &usernames, bool first) {
  HttpRequest *request = new HttpRequest(this, false);
  QObject::connect(request, &HttpRequest::requestDone, [=](const QString &resp, bool success) {
    if (success) {
      if (!resp.isEmpty()) {
        std::string keys = resp.toStdString();
        if (first) {
          params.put("GithubUsername", usernames.join(",").toStdString());
        } else {
          // Add accumulated user keys
          keys += "\n" + params.get("GithubSshKeys");
        }
        params.put("GithubSshKeys", keys);
      } else {
        ConfirmationDialog::alert(tr("Username '%1' has no keys on GitHub").arg(usernames.at(0)), this);
      }

      // Get next username if 1 left
      if (usernames.size() > 1) {
        getUserKeys(usernames.mid(1, usernames.size() - 1), false);
      }
    } else {
      clearParams();
      if (request->timeout()) {
        ConfirmationDialog::alert(tr("Request timed out"), this);
      } else {
        ConfirmationDialog::alert(tr("Username '%1' doesn't exist on GitHub").arg(usernames.at(0)), this);
      }
    }

    refresh();
    request->deleteLater();
  });

  if (usernames.size() > 0) {
    request->sendRequest("https://github.com/" + usernames.at(0).trimmed() + ".keys");
  }
}
