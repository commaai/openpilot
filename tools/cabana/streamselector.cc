#include "tools/cabana/streamselector.h"

#include <QFileDialog>
#include <QLabel>
#include <QPushButton>

#include "streams/socketcanstream.h"
#include "tools/cabana/streams/devicestream.h"
#include "tools/cabana/streams/pandastream.h"
#include "tools/cabana/streams/replaystream.h"
#include "tools/cabana/streams/socketcanstream.h"

StreamSelector::StreamSelector(AbstractStream **stream, QWidget *parent) : QDialog(parent) {
  setWindowTitle(tr("Open stream"));
  QVBoxLayout *layout = new QVBoxLayout(this);
  tab = new QTabWidget(this);
  layout->addWidget(tab);

  QHBoxLayout *dbc_layout = new QHBoxLayout();
  dbc_file = new QLineEdit(this);
  dbc_file->setReadOnly(true);
  dbc_file->setPlaceholderText(tr("Choose a dbc file to open"));
  QPushButton *file_btn = new QPushButton(tr("Browse..."));
  dbc_layout->addWidget(new QLabel(tr("dbc File")));
  dbc_layout->addWidget(dbc_file);
  dbc_layout->addWidget(file_btn);
  layout->addLayout(dbc_layout);

  QFrame *line = new QFrame(this);
  line->setFrameStyle(QFrame::HLine | QFrame::Sunken);
  layout->addWidget(line);

  btn_box = new QDialogButtonBox(QDialogButtonBox::Open | QDialogButtonBox::Cancel);
  layout->addWidget(btn_box);

  addStreamWidget(ReplayStream::widget(stream));
  addStreamWidget(PandaStream::widget(stream));
  if (SocketCanStream::available()) {
    addStreamWidget(SocketCanStream::widget(stream));
  }
  addStreamWidget(DeviceStream::widget(stream));

  QObject::connect(btn_box, &QDialogButtonBox::rejected, this, &QDialog::reject);
  QObject::connect(btn_box, &QDialogButtonBox::accepted, [=]() {
    setEnabled(false);
    if (((AbstractOpenStreamWidget *)tab->currentWidget())->open()) {
      accept();
    }
    setEnabled(true);
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
  auto open_btn = btn_box->button(QDialogButtonBox::Open);
  QObject::connect(w, &AbstractOpenStreamWidget::enableOpenButton, open_btn, &QPushButton::setEnabled);
}
