#include "selfdrive/ui/qt/widgets/external_storage.h"

#include <QProcess>
#include <QCoreApplication>
#include <QShowEvent>
#include <QTimer>

#include "common/params.h"
#include "selfdrive/ui/qt/api.h"
#include "selfdrive/ui/qt/widgets/input.h"

ExternalStorageControl::ExternalStorageControl() :
  ButtonControl(tr("External Storage"), "", tr("Extend your comma device's storage by inserting a USB drive into the aux port.")) {

  QObject::connect(this, &ButtonControl::clicked, [=]() {
    if (text() == tr("MOUNT")) {
      setText(tr("mounting"));
      setEnabled(false);
      mountStorage();
    } else if (text() == tr("UNMOUNT")) {
      setText(tr("unmounting"));
      setEnabled(false);
      unmountStorage();
    } else if (text() == tr("FORMAT")) {
      if (ConfirmationDialog::confirm(tr("Are you sure you want to format this drive? This will erase all data."), tr("Format"), this)) {
        setText(tr("formatting"));
        setEnabled(false);
        formatStorage();
      }
    }
  });
  refresh();
}

void ExternalStorageControl::refresh() {
  bool isMounted = isStorageMounted();
  bool hasFilesystem = isFilesystemPresent();
  QString storageInfo = "";

  if (isMounted) {
    QProcess process;
    process.start("sh", QStringList() << "-c" << "df -h /mnt/external_realdata | awk 'NR==2 {print $3 \"/\" $2}'");
    process.waitForFinished();
    storageInfo = process.readAllStandardOutput().trimmed();
  }

  if (!hasFilesystem || !isDriveInitialized()) {
    setValue(tr("needs format"));
    setText(tr("FORMAT"));
  } else if (isMounted) {
    setValue(storageInfo);
    setText(tr("UNMOUNT"));
  } else {
    setValue(tr(""));
    setText(tr("MOUNT"));
  }

  setEnabled(true);
}

bool ExternalStorageControl::isStorageMounted() {
  QProcess process;
  process.start("sh", QStringList() << "-c" << "findmnt -n /mnt/external_realdata");
  process.waitForFinished();
  
  QString output = process.readAllStandardOutput().trimmed();
  return !output.isEmpty();
}

bool ExternalStorageControl::isFilesystemPresent() {
  QProcess process;
  process.start("sh", QStringList() << "-c" << "lsblk -f /dev/sdg1 | grep -q ext4");
  process.waitForFinished();
  return process.exitCode() == 0;
}

bool ExternalStorageControl::isDriveInitialized() {
  QProcess process;
  process.start("sh", QStringList() << "-c" << "sudo blkid /dev/sdg1 | grep -q 'LABEL=\"openpilot\"'");
  process.waitForFinished();
  return process.exitCode() == 0;
}

void ExternalStorageControl::mountStorage() {
  QProcess::execute("sh", QStringList() << "-c" << "sudo mkdir -p /mnt/external_realdata");
  setText(tr("mounting"));
  setEnabled(false);
  QCoreApplication::processEvents();

  QProcess process;
  process.start("sh", QStringList() << "-c" << "sudo mount /mnt/external_realdata");
  process.waitForFinished();

  refresh();
}

void ExternalStorageControl::unmountStorage() {
  setText(tr("unmounting"));
  setEnabled(false);
  QCoreApplication::processEvents();

  QProcess process;
  process.start("sh", QStringList() << "-c" << "sudo umount /mnt/external_realdata");
  process.waitForFinished();

  refresh();
}

void ExternalStorageControl::formatStorage() {
  unmountStorage();

  setText(tr("formatting"));
  setEnabled(false);
  QCoreApplication::processEvents();

  QProcess *process = new QProcess(this);
  connect(process, QOverload<int, QProcess::ExitStatus>::of(&QProcess::finished),
          this, [=](int exitCode, QProcess::ExitStatus exitStatus) {
            if (exitCode == 0 && exitStatus == QProcess::NormalExit) {
              checkAndUpdateFstab();
              mountStorage();
              QProcess::execute("sh", QStringList() << "-c" << "sudo chown -R $(whoami):$(whoami) /mnt/external_realdata");
              QProcess::execute("sh", QStringList() << "-c" << "sudo chmod -R 775 /mnt/external_realdata");
              QProcess::execute("sh", QStringList() << "-c" << "sudo e2label /dev/sdg1 openpilot");
            } else {
              qWarning() << "Formatting failed with exit code" << exitCode;
            }
            process->deleteLater();
            refresh();
          });

  process->start("sh", QStringList() << "-c" << "sudo mkfs.ext4 -F /dev/sdg1");
}

void ExternalStorageControl::checkAndUpdateFstab() {
  QString fstabEntry = "/dev/sdg1 /mnt/external_realdata ext4 defaults,nofail 0 2";
  QProcess checkProcess;
  checkProcess.start("sh", QStringList() << "-c" << "grep -Fxq '" + fstabEntry + "' /etc/fstab");
  checkProcess.waitForFinished();

  if (checkProcess.exitCode() != 0) {
    QProcess::execute("sh", QStringList() << "-c" << "sudo mount -o remount,rw /");
    QProcess::execute("sh", QStringList() << "-c" << "echo '" + fstabEntry + "' | sudo tee -a /etc/fstab");
    QProcess::execute("sh", QStringList() << "-c" << "sudo systemctl daemon-reload");
    QProcess::execute("sh", QStringList() << "-c" << "sudo mount -o remount,ro /");
  }
}

void ExternalStorageControl::showEvent(QShowEvent *event) {
  ButtonControl::showEvent(event);
  QTimer::singleShot(100, this, &ExternalStorageControl::refresh);
}
