#include <QApplication>
#include <QHBoxLayout>
#include <QLabel>
#include <QPushButton>
#include <QVBoxLayout>
#include <QWidget>

#include "selfdrive/ui/qt/qt_window.h"

#define USERDATA "/dev/disk/by-partlabel/userdata"
#define NVME "/dev/nvme0n1"

bool do_reset() {
  std::vector<const char*> cmds = {
    "sudo umount " NVME,
    "yes | sudo mkfs.ext4 " NVME,
    "sudo umount " USERDATA,
    "yes | sudo mkfs.ext4 " USERDATA,
    "sudo reboot",
  };

  for (auto &cmd : cmds) {
    int ret = std::system(cmd);
    if (ret != 0) return false;
  }
  return true;
}

int main(int argc, char *argv[]) {
  QApplication a(argc, argv);
  QWidget window;
  setMainWindow(&window);

  QVBoxLayout *main_layout = new QVBoxLayout(&window);
  main_layout->setContentsMargins(125, 125, 125, 125);

  QLabel *title = new QLabel("System Reset");
  title->setStyleSheet(R"(
    font-weight: 500;
    font-size: 100px;
  )");
  main_layout->addWidget(title, 0, Qt::AlignTop);

  QLabel *body = new QLabel("System reset triggered. Press confirm to erase all content and settings. Press cancel to resume boot.");
  body->setWordWrap(true);
  body->setAlignment(Qt::AlignCenter);
  body->setStyleSheet("font-size: 65px;");
  main_layout->addWidget(body, 1, Qt::AlignCenter);

  QHBoxLayout *btn_layout = new QHBoxLayout();

  QPushButton *cancel_btn = new QPushButton("Cancel");
  btn_layout->addWidget(cancel_btn, 0, Qt::AlignLeft);
  QObject::connect(cancel_btn, &QPushButton::released, &a, &QApplication::quit);

  QPushButton *confirm_btn  = new QPushButton("Confirm");
  btn_layout->addWidget(confirm_btn, 0, Qt::AlignRight);
  QObject::connect(confirm_btn, &QPushButton::released, [=]() {
    const QString confirm_txt = "Are you sure you want to reset your device?";
    if (body->text() != confirm_txt) {
      body->setText(confirm_txt);
    } else {
      body->setText("Resetting device...");
      cancel_btn->hide();
      confirm_btn->hide();
      QCoreApplication::processEvents(QEventLoop::AllEvents, 1000);
#ifdef __aarch64__
      bool ret = do_reset();
      if (!ret) {
        body->setText("Reset failed.");
        cancel_btn->show();
      }
#endif
    }
  });

  main_layout->addLayout(btn_layout);

  window.setStyleSheet(R"(
    * {
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

  return a.exec();
}
