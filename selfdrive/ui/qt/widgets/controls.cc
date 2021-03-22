#include "controls.hpp"

 QFrame *horizontal_line(QWidget *parent) {
  QFrame *line = new QFrame(parent);
  line->setFrameShape(QFrame::StyledPanel);
  line->setStyleSheet("margin-left: 40px; margin-right: 40px; border-width: 1px; border-bottom-style: solid; border-color: gray;");
  line->setFixedHeight(2);
  return line;
}

AbstractControl::AbstractControl(const QString &title, const QString &desc, const QString &icon) : QFrame() {
  hboxLayout = new QHBoxLayout;
  hboxLayout->setSpacing(50);
  // left icon
  if (!icon.isEmpty()) {
    QPixmap pix(icon);
    QLabel *icon = new QLabel();
    icon->setPixmap(pix.scaledToWidth(80, Qt::SmoothTransformation));
    icon->setSizePolicy(QSizePolicy(QSizePolicy::Fixed, QSizePolicy::Fixed));
    hboxLayout->addWidget(icon);
  }
  // title
  title_label = new QLabel(title);
  title_label->setStyleSheet("font-size: 50px;");
  hboxLayout->addWidget(title_label);

  QVBoxLayout *vboxLayout = new QVBoxLayout(this);
  vboxLayout->setContentsMargins(0, 0, 0, 0);
  vboxLayout->addLayout(hboxLayout);
  
  // description
  if (!desc.isEmpty()) {
    desc_label = new QLabel(desc);
    desc_label->setContentsMargins(40, 0, 40, 0);
    desc_label->setStyleSheet("font-size: 40px;color:grey");
    desc_label->setWordWrap(true);
    desc_label->setVisible(false);
    vboxLayout->addStretch();
    vboxLayout->addWidget(desc_label);
  }
}
