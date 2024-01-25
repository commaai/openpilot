#include <QApplication>
#include <QHBoxLayout>
#include <QLabel>
#include <QPushButton>
#include <QTimer>
#include <QVBoxLayout>

#include "selfdrive/ui/qt/qt_window.h"
#include "selfdrive/ui/qt/setup/reset.h"

#define NVME "/dev/nvme0n1"
#define USERDATA "/dev/disk/by-partlabel/userdata"

void Reset::doErase() {
  // best effort to wipe nvme
  std::system("sudo umount " NVME);
  std::system("yes | sudo mkfs.ext4 " NVME);

  int rm = std::system("sudo rm -rf /data/*");
  std::system("sudo umount " USERDATA);
  int fmt = std::system("yes | sudo mkfs.ext4 " USERDATA);

  if (rm == 0 || fmt == 0) {
    std::system("sudo reboot");
  }
  body->setText(tr("Reset failed. Reboot to try again."));
  rebootBtn->show();
}

void Reset::startReset() {
  body->setText(tr("Resetting device...\nThis may take up to a minute."));
  rejectBtn->hide();
  rebootBtn->hide();
  confirmBtn->hide();
#ifdef __aarch64__
  QTimer::singleShot(100, this, &Reset::doErase);
#endif
}

void Reset::confirm() {
  const QString confirm_txt = tr("Are you sure you want to reset your device?");
  if (body->text() != confirm_txt) {
    body->setText(confirm_txt);
  } else {
    startReset();
  }
}

Reset::Reset(ResetMode mode, QWidget *parent) : QWidget(parent) {
  QVBoxLayout *main_layout = new QVBoxLayout(this);
  main_layout->setContentsMargins(45, 220, 45, 45);
  main_layout->setSpacing(0);

  QLabel *title = new QLabel(tr("System Reset"));
  title->setStyleSheet("font-size: 90px; font-weight: 600;");
  main_layout->addWidget(title, 0, Qt::AlignTop | Qt::AlignLeft);

  main_layout->addSpacing(60);

  body = new QLabel(tr("System reset triggered. Press confirm to erase all content and settings. Press cancel to resume boot."));
  body->setWordWrap(true);
  body->setStyleSheet("font-size: 80px; font-weight: light;");
  main_layout->addWidget(body, 1, Qt::AlignTop | Qt::AlignLeft);

  QHBoxLayout *blayout = new QHBoxLayout();
  main_layout->addLayout(blayout);
  blayout->setSpacing(50);

  rejectBtn = new QPushButton(tr("Cancel"));
  blayout->addWidget(rejectBtn);
  QObject::connect(rejectBtn, &QPushButton::clicked, QCoreApplication::instance(), &QCoreApplication::quit);

  rebootBtn = new QPushButton(tr("Reboot"));
  blayout->addWidget(rebootBtn);
#ifdef __aarch64__
  QObject::connect(rebootBtn, &QPushButton::clicked, [=]{
    std::system("sudo reboot");
  });
#endif

  confirmBtn = new QPushButton(tr("Confirm"));
  confirmBtn->setStyleSheet(R"(
    QPushButton {
      background-color: #465BEA;
    }
    QPushButton:pressed {
      background-color: #3049F4;
    }
  )");
  blayout->addWidget(confirmBtn);
  QObject::connect(confirmBtn, &QPushButton::clicked, this, &Reset::confirm);

  bool recover = mode == ResetMode::RECOVER;
  rejectBtn->setVisible(!recover);
  rebootBtn->setVisible(recover);
  if (recover) {
    body->setText(tr("Unable to mount data partition. Partition may be corrupted. Press confirm to erase and reset your device."));
  }

  setStyleSheet(R"(
    * {
      font-family: Inter;
      color: white;
      background-color: black;
    }
    QLabel {
      margin-left: 140;
    }
    QPushButton {
      height: 160;
      font-size: 55px;
      font-weight: 400;
      border-radius: 10px;
      background-color: #333333;
    }
    QPushButton:pressed {
      background-color: #444444;
    }
  )");
}

int main(int argc, char *argv[]) {
  ResetMode mode = ResetMode::USER_RESET;
  if (argc > 1) {
    if (strcmp(argv[1], "--recover") == 0) {
      mode = ResetMode::RECOVER;
    }
  }

  QApplication a(argc, argv);
  Reset reset(mode);
  setMainWindow(&reset);
  return a.exec();
}
