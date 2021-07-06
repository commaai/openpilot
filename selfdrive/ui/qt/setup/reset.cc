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

void Reset::doReset() {
  std::vector<const char*> cmds = {
    "sudo umount " NVME " || true",
    "yes | sudo mkfs.ext4 " NVME " || true",
    "sudo umount " USERDATA " || true",
    "yes | sudo mkfs.ext4 " USERDATA,
    "sudo reboot",
  };

  for (auto &cmd : cmds) {
    int ret = std::system(cmd);
    if (ret != 0) {
      body->setText("Reset failed. Reboot to try again.");
      rebootBtn->show();
      return;
    }
  }
}

void Reset::confirm() {
  const QString confirm_txt = "Are you sure you want to reset your device?";
  if (body->text() != confirm_txt) {
    body->setText(confirm_txt);
  } else {
    body->setText("Resetting device...");
    rejectBtn->hide();
    rebootBtn->hide();
    confirmBtn->hide();
#ifdef __aarch64__
    QTimer::singleShot(100, this, &Reset::doReset);
#endif
  }
}

Reset::Reset(bool recover, QWidget *parent) : QWidget(parent) {
  QVBoxLayout *main_layout = new QVBoxLayout(this);
  main_layout->setContentsMargins(100, 100, 100, 100);

  QLabel *title = new QLabel("System Reset");
  title->setStyleSheet(R"(font-weight: 500; font-size: 100px;)");
  main_layout->addWidget(title, 0, Qt::AlignTop);

  body = new QLabel("System reset triggered. Press confirm to erase all content and settings. Press cancel to resume boot.");
  body->setWordWrap(true);
  body->setAlignment(Qt::AlignCenter);
  body->setStyleSheet("font-size: 80px;");
  main_layout->addWidget(body, 1, Qt::AlignCenter);

  QHBoxLayout *blayout = new QHBoxLayout();
  main_layout->addLayout(blayout);

  rejectBtn = new QPushButton("Cancel");
  blayout->addWidget(rejectBtn, 0, Qt::AlignLeft);
  QObject::connect(rejectBtn, &QPushButton::released, QCoreApplication::instance(), &QCoreApplication::quit);

  rebootBtn = new QPushButton("Reboot");
  blayout->addWidget(rebootBtn, 0, Qt::AlignLeft);
  QObject::connect(rebootBtn, &QPushButton::released, [=]{
    std::system("sudo reboot");
  });

  confirmBtn  = new QPushButton("Confirm");
  blayout->addWidget(confirmBtn, 0, Qt::AlignRight);
  QObject::connect(confirmBtn, &QPushButton::released, this, &Reset::confirm);

  rejectBtn->setVisible(!recover);
  rebootBtn->setVisible(recover);
  if (recover) {
    body->setText("Unable to mount data partition. Press confirm to reset your device.");
  }

  setStyleSheet(R"(
    * {
      font-family: Inter;
      color: white;
      background-color: black;
    }
    QPushButton {
      padding: 50px;
      padding-right: 100px;
      padding-left: 100px;
      border: 7px solid white;
      border-radius: 20px;
      font-size: 50px;
    }
  )");
}

int main(int argc, char *argv[]) {
  bool recover = argc > 1 && strcmp(argv[1], "--recover") == 0;
  QApplication a(argc, argv);
  Reset reset(recover);
  setMainWindow(&reset);
  return a.exec();
}
