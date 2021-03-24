#include "controls.hpp"

QFrame *horizontal_line(QWidget *parent) {
  QFrame *line = new QFrame(parent);
  line->setFrameShape(QFrame::StyledPanel);
  line->setStyleSheet(R"(
    margin-left: 40px;
    margin-right: 40px;
    border-width: 1px;
    border-bottom-style: solid;
    border-color: gray;
  )");
  line->setFixedHeight(2);
  return line;
}

AbstractControl::AbstractControl(const QString &title, const QString &desc, const QString &icon, QWidget *parent) : QFrame(parent) {
  QVBoxLayout *vlayout = new QVBoxLayout();

  hlayout = new QHBoxLayout;
  hlayout->setMargin(0);
  hlayout->setSpacing(50);

  // left icon
  if (!icon.isEmpty()) {
    QPixmap pix(icon);
    QLabel *icon = new QLabel();
    icon->setPixmap(pix.scaledToWidth(80, Qt::SmoothTransformation));
    icon->setSizePolicy(QSizePolicy(QSizePolicy::Fixed, QSizePolicy::Fixed));
    hlayout->addWidget(icon);
  }

  // title
  title_label = new QPushButton(title);
  title_label->setStyleSheet("font-size: 50px; font-weight: 400; text-align: left; background: none;");
  hlayout->addWidget(title_label);

  vlayout->addLayout(hlayout);

  // description
  if (!desc.isEmpty()) {
    description = new QLabel(desc);
    description->setContentsMargins(40, 20, 40, 20);
    description->setStyleSheet("font-size: 40px; color:grey");
    description->setWordWrap(true);
    description->setVisible(false);
    vlayout->addWidget(description);

    connect(title_label, &QPushButton::clicked, [=]() {
      description->setVisible(!description->isVisible());
    });
  }

  setLayout(vlayout);
}
