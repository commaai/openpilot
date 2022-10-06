#include "tools/cabana/mainwin.h"

#include <QHBoxLayout>
#include <QVBoxLayout>

MainWindow::MainWindow() : QWidget() {
  QVBoxLayout *main_layout = new QVBoxLayout(this);

  QHBoxLayout *h_layout = new QHBoxLayout();
  main_layout->addLayout(h_layout);

  messages_widget = new MessagesWidget(this);
  h_layout->addWidget(messages_widget);

  detail_widget = new DetailWidget(this);
  detail_widget->setFixedWidth(600);
  h_layout->addWidget(detail_widget);

  // right widgets
  QWidget *right_container = new QWidget(this);
  right_container->setFixedWidth(640);
  r_layout = new QVBoxLayout(right_container);

  video_widget = new VideoWidget(this);
  r_layout->addWidget(video_widget, 0, Qt::AlignTop);

  charts_widget = new ChartsWidget(this);
  r_layout->addWidget(charts_widget);

  h_layout->addWidget(right_container);

  QObject::connect(messages_widget, &MessagesWidget::msgChanged, detail_widget, &DetailWidget::setMsg);
  QObject::connect(charts_widget, &ChartsWidget::floatingCharts, this, &MainWindow::floatingCharts);
}

void MainWindow::floatingCharts(bool floating) {
  if (floating && !floating_window) {
    floating_window = new FloatWindow(nullptr);
    floating_window->layout()->addWidget(charts_widget);
    floating_window->showMaximized();
    QObject::connect(floating_window, &FloatWindow::closing, [this]() { floatingCharts(false); });
  } else if (!floating && floating_window) {
    r_layout->addWidget(charts_widget);
    floating_window->deleteLater();
    floating_window = nullptr;
  }
}

void MainWindow::closeEvent(QCloseEvent *event) {
  if (floating_window)
    floating_window->deleteLater();
  QWidget::closeEvent(event);
}

// FloatWindow

FloatWindow::FloatWindow(QWidget *parent) : QWidget(parent) {
  setLayout(new QVBoxLayout());
}

void FloatWindow::closeEvent(QCloseEvent *event) {
  emit closing();
  QWidget::closeEvent(event);
}
