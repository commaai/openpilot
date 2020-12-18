#include <QLabel>
#include <QWidget>
#include <QPushButton>
#include <QVBoxLayout>
#include <QApplication>

#include "qt_window.hpp"

int main(int argc, char *argv[]) {
  QApplication a(argc, argv);
  QWidget window;
  setMainWindow(&window);

  QVBoxLayout *layout = new QVBoxLayout();
  layout->setContentsMargins(125, 125, 125, 125);

  // TODO: make this scroll
  layout->addWidget(new QLabel(argv[1]), 0, Qt::AlignTop);

  QPushButton *btn = new QPushButton();
#ifdef __aarch64__
  btn->setText("Reboot");
  QObject::connect(btn, &QPushButton::released, [=]() {
    std::system("sudo reboot");
  });
#else
  btn->setText("Exit");
  QObject::connect(btn, SIGNAL(released()), &a, SLOT(quit()));
#endif
  layout->addWidget(btn, 0, Qt::AlignRight);

  window.setLayout(layout);
  window.setStyleSheet(R"(
    * {
      color: white;
      background-color: black;
      font-size: 60px;
    }
    QPushButton {
      padding: 50px;
      padding-right: 100px;
      padding-left: 100px;
      border: 2px solid white;
      border-radius: 20px;
    }
  )");

  return a.exec();
}
