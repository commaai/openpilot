#include "tools/cabana/mainwin.h"

#include <QHBoxLayout>
#include <QVBoxLayout>

MainWindow::MainWindow() : QWidget() {
  assert(parser != nullptr);

  QVBoxLayout *main_layout = new QVBoxLayout(this);
  QHBoxLayout *h_layout = new QHBoxLayout();
  main_layout->addLayout(h_layout);

  messages_widget = new MessagesWidget(this);
  QObject::connect(messages_widget, &MessagesWidget::msgChanged, [=](const QString &id) {
    detail_widget->setMsg(id);
  });
  h_layout->addWidget(messages_widget);

  detail_widget = new DetailWidget(this);
  h_layout->addWidget(detail_widget);

  // right widget
  QWidget *right_container = new QWidget(this);
  right_container->setFixedWidth(640);
  QVBoxLayout *r_layout = new QVBoxLayout(right_container);
  video_widget = new VideoWidget(this);
  r_layout->addWidget(video_widget);

  charts_widget = new ChartsWidget(this);
  QScrollArea *scroll = new QScrollArea(this);
  scroll->setWidget(charts_widget);
  scroll->setWidgetResizable(true);
  scroll->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
  scroll->setSizePolicy(QSizePolicy::Preferred, QSizePolicy::Expanding);
  r_layout->addWidget(scroll);

  h_layout->addWidget(right_container);
}
