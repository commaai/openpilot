#include "tools/cabana/streamselector.h"

#include <QDialogButtonBox>
#include <QFileDialog>
#include <QFormLayout>
#include <QLabel>
#include <QPushButton>

StreamSelector::StreamSelector(QWidget *parent) : QDialog(parent) {
  setWindowTitle(tr("Open stream"));
  QVBoxLayout *main_layout = new QVBoxLayout(this);

  tab = new QTabWidget(this);
  tab->setTabBarAutoHide(true);
  main_layout->addWidget(tab);

  QHBoxLayout *dbc_layout = new QHBoxLayout();
  dbc_file = new QLineEdit(this);
  dbc_file->setReadOnly(true);
  dbc_file->setPlaceholderText(tr("Choose a dbc file to open"));
  QPushButton *file_btn = new QPushButton(tr("Browse..."));
  dbc_layout->addWidget(new QLabel(tr("dbc File")));
  dbc_layout->addWidget(dbc_file);
  dbc_layout->addWidget(file_btn);
  main_layout->addLayout(dbc_layout);

  QFrame *line = new QFrame(this);
  line->setFrameStyle(QFrame::HLine | QFrame::Sunken);
  main_layout->addWidget(line);

  auto btn_box = new QDialogButtonBox(QDialogButtonBox::Open | QDialogButtonBox::Cancel);
  main_layout->addWidget(btn_box);

  QObject::connect(btn_box, &QDialogButtonBox::rejected, this, &QDialog::reject);
  QObject::connect(btn_box, &QDialogButtonBox::accepted, [=]() {
    success = ((AbstractOpenStreamWidget *)tab->currentWidget())->open();
    if (success) {
      accept();
    }
  });
  QObject::connect(file_btn, &QPushButton::clicked, [this]() {
    QString fn = QFileDialog::getOpenFileName(this, tr("Open File"), settings.last_dir, "DBC (*.dbc)");
    if (!fn.isEmpty()) {
      dbc_file->setText(fn);
      settings.last_dir = QFileInfo(fn).absolutePath();
    }
  });
}

void StreamSelector::addStreamWidget(AbstractOpenStreamWidget *w) {
  tab->addTab(w, w->title());
}
