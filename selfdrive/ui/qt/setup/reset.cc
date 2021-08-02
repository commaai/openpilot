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
  main_layout->setContentsMargins(45, 220, 45, 45);
  main_layout->setSpacing(0);

  QLabel *title = new QLabel("System Reset");
  title->setStyleSheet("font-size: 90px; font-weight: 600;");
  main_layout->addWidget(title, 0, Qt::AlignTop | Qt::AlignLeft);

  main_layout->addSpacing(60);

  body = new QLabel("System reset triggered. Press confirm to erase all content and settings. Press cancel to resume boot.");
  body->setWordWrap(true);
  body->setStyleSheet("font-size: 80px; font-weight: light;");
  main_layout->addWidget(body, 1, Qt::AlignTop | Qt::AlignLeft);

  QHBoxLayout *blayout = new QHBoxLayout();
  main_layout->addLayout(blayout);
  blayout->setSpacing(50);

  rejectBtn = new QPushButton("Cancel");
  blayout->addWidget(rejectBtn);
  QObject::connect(rejectBtn, &QPushButton::clicked, QCoreApplication::instance(), &QCoreApplication::quit);

  rebootBtn = new QPushButton("Reboot");
  blayout->addWidget(rebootBtn);
#ifdef __aarch64__
  QObject::connect(rebootBtn, &QPushButton::clicked, [=]{
    std::system("sudo reboot");
  });
#endif

  confirmBtn = new QPushButton("Confirm");
  confirmBtn->setStyleSheet("background-color: #465BEA;");
  blayout->addWidget(confirmBtn);
  QObject::connect(confirmBtn, &QPushButton::clicked, this, &Reset::confirm);

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
  )");
}

int main(int argc, char *argv[]) {
  bool recover = argc > 1 && strcmp(argv[1], "--recover") == 0;
  QApplication a(argc, argv);
  Reset reset(recover);
  setMainWindow(&reset);
  return a.exec();
}
