#include "selfdrive/ui/qt/widgets/secockey_restore.h"

#include "common/params.h"
#include "selfdrive/ui/qt/api.h"
#include "selfdrive/ui/qt/widgets/input.h"

SecOCKeyRestore::SecOCKeyRestore() :
  ButtonControl(tr("Restore SecOC Key"), tr("RESTORE"), "") {

  QObject::connect(this, &ButtonControl::clicked, [=]() {
    setEnabled(false);

    QString installed = QString::fromStdString(params.get("SecOCKey"));
    QString archived = getArchive("/cache/params/SecOCKey");

    if (isValid(installed)) {
      if (isValid(archived) && archived != installed) {
        bool overwrite = ConfirmationDialog(
            tr("Installed: %1\n       New: %2\n\nInstall the new security key?").arg(installed, archived),
            "Install", tr("Cancel"), true, this
        ).exec();
        if (overwrite) {
          params.put("SecOCKey", archived.toStdString());
        }
      } else {
        ConfirmationDialog::alert(tr("Security key already installed\n%1").arg(installed), this);
      }
    } else {
      if (isValid(archived)) {
        // Happy path
        ConfirmationDialog::alert(tr("Security key restored and installed\n%1").arg(installed), this);
      } else {
        ConfirmationDialog::alert(tr("New security key not found"), this);
      }
    }

    refresh(); // Live update
  });

  refresh(); // Initial draw
}

void SecOCKeyRestore::refresh() {
  QString key = QString::fromStdString(params.get("SecOCKey"));
  if (!key.length()) {
    key = tr("Not Installed");
  }
  setDescription(key);
  setEnabled(true);
}

QString SecOCKeyRestore::getArchive(QString filePath) {
  QFile archiveFile(filePath);

  // Archived key file doesn't exist
  if (!archiveFile.open(QIODevice::ReadOnly | QIODevice::Text)) {
    return "";
  }

  QTextStream in(&archiveFile);
  QString key = in.readAll();
  archiveFile.close();

  // Archived key file can't be read
  if (in.status() != QTextStream::Ok) {
    return "";
  }

  // Archived key is not a valid key
  if (!isValid(key)) {
    return "";
  }

  // Return the key
  return key;
}

// Check if the key is a 32 characters long hexadecimal string
bool SecOCKeyRestore::isValid(QString key) {
  if (key.length() != 32) {
    return false;
  }

  // Check if each character is a valid hexadecimal digit (0-9, a-f)
  for (QChar c : key) {
    if (!c.isDigit() && !(c.isLower() && c >= 'a' && c <= 'f')) {
      return false;
    }
  }

  return true;
}
