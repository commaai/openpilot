#include <QLabel>
#include <QWidget>
#include <QPushButton>
#include <QHBoxLayout>
#include <QVBoxLayout>
#include <QApplication>

#include "qt_window.hpp"


void do_reset() {
  // format userdata

  // reboot
  std::system("sudo reboot");
}

int main(int argc, char *argv[]) {
  QApplication a(argc, argv);
  QWidget window;
  setMainWindow(&window);

  QVBoxLayout *layout = new QVBoxLayout();
  layout->setContentsMargins(125, 125, 125, 125);

  // TODO: make this scroll
  QLabel *title = new QLabel("System Reset");
  title->setStyleSheet(R"(
    font-size: 100px;
    font-weight: bold;
  )");
  layout->addWidget(title, 0, Qt::AlignTop);

  QLabel *body = new QLabel("Factory reset triggered. Press confirm to erase all content and settings. Press cancel to resume boot.");
  body->setWordWrap(true);
  body->setStyleSheet(R"(
    font-size: 65px;
  )");
  layout->addWidget(body, 1, Qt::AlignCenter);

  QHBoxLayout *btn_layout = new QHBoxLayout();

  QPushButton *cancel_btn = new QPushButton("Cancel");
  btn_layout->addWidget(cancel_btn, 0, Qt::AlignLeft);
  QObject::connect(cancel_btn, &QPushButton::released, [=]() {  });

  QPushButton *confirm_btn  = new QPushButton("Confirm");
  btn_layout->addWidget(confirm_btn, 0, Qt::AlignRight);
#ifdef __aarch64__
  QObject::connect(confirm_btn, &QPushButton::released, [=]() { do_reset(); });
#endif

  layout->addLayout(btn_layout);

  window.setLayout(layout);
  window.setStyleSheet(R"(
    * {
      color: white;
      background-color: black;
    }
    QPushButton {
      padding: 50px;
      padding-right: 100px;
      padding-left: 100px;
      border: 2px solid white;
      border-radius: 20px;
      font-size: 50px;
    }
  )");

  return a.exec();
}
