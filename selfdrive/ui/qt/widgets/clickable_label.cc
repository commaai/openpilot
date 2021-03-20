#include "clickable_label.hpp"

#include <QHBoxLayout>
#include <QVBoxLayout>

static QFrame *horizontal_line(QWidget *parent = 0) {
  QFrame *line = new QFrame(parent);
  line->setFrameShape(QFrame::StyledPanel);
  line->setStyleSheet("margin-left: 40px; margin-right: 40px; border-width: 1px; border-bottom-style: solid; border-color: gray;");
  line->setFixedHeight(2);
  return line;
}

ClickableLabel::ClickableLabel(const QString &title, const QString &desc, QWidget *control, const QString &icon, bool bottom_line) : QFrame() {
  QHBoxLayout *layout = new QHBoxLayout;
  layout->setContentsMargins(0, 15, 0, 15);
  layout->setSpacing(50);

  // left icon
  if (!icon.isEmpty()) {
    QPixmap pix(icon);
    QLabel *icon = new QLabel();
    icon->setPixmap(pix.scaledToWidth(80, Qt::SmoothTransformation));
    icon->setSizePolicy(QSizePolicy(QSizePolicy::Fixed, QSizePolicy::Fixed));
    layout->addWidget(icon);
  }
  // title
  QLabel *title_label = new QLabel(title);
  title_label->setStyleSheet(R"(font-size: 50px;)");
  layout->addWidget(title_label);

  // right control
  layout->addWidget(control);

  QVBoxLayout *main_l = new QVBoxLayout(this);
  main_l->addLayout(layout);
  main_l->addStretch();
  // description
  if (!desc.isEmpty()) {
    desc_label = new QLabel(desc);
    desc_label->setContentsMargins(40, 15, 40, 15);
    desc_label->setStyleSheet(R"(font-size: 40px;color:grey)");
    desc_label->setWordWrap(true);
    desc_label->setVisible(false);
    main_l->addWidget(desc_label);
  }

  if (bottom_line) {
    main_l->addWidget(horizontal_line());
  }
}
