#include "tools/cabana/tools/findsimilarbits.h"

#include <QGridLayout>
#include <QHeaderView>
#include <QHBoxLayout>
#include <QIntValidator>
#include <QLabel>
#include <QPushButton>
#include <QRadioButton>

#include "tools/cabana/dbcmanager.h"
#include "tools/cabana/streams/abstractstream.h"

FindSimilarBitsDlg::FindSimilarBitsDlg(QWidget *parent) : QDialog(parent, Qt::WindowFlags() | Qt::Window) {
  setWindowTitle(tr("Find similar bits"));
  setAttribute(Qt::WA_DeleteOnClose);

  QVBoxLayout *main_layout = new QVBoxLayout(this);

  QHBoxLayout *src_layout = new QHBoxLayout();
  src_bus_combo = new QComboBox(this);
  find_bus_combo = new QComboBox(this);
  SourceSet bus_set;
  for (auto it = can->last_msgs.begin(); it != can->last_msgs.end(); ++it) {
    bus_set << it.key().source;
  }
  for (auto cb : {src_bus_combo, find_bus_combo}) {
    for (uint8_t bus : bus_set) {
      cb->addItem(QString::number(bus), bus);
    }
    cb->model()->sort(0);
    cb->setCurrentIndex(0);
  }

  msg_cb = new QComboBox(this);
  // TODO: update when src_bus_combo changes
  for (auto &[id, msg] : dbc()->getMessages(0)) {
    msg_cb->addItem(msg.name, id.address);
  }
  msg_cb->model()->sort(0);
  msg_cb->setCurrentIndex(0);

  byte_idx_sb = new QSpinBox(this);
  byte_idx_sb->setFixedWidth(50);
  byte_idx_sb->setRange(0, 63);

  bit_idx_sb = new QSpinBox(this);
  bit_idx_sb->setFixedWidth(50);
  bit_idx_sb->setRange(0, 7);

  src_layout->addWidget(new QLabel(tr("Bus")));
  src_layout->addWidget(src_bus_combo);
  src_layout->addWidget(msg_cb);
  src_layout->addWidget(new QLabel(tr("Byte Index")));
  src_layout->addWidget(byte_idx_sb);
  src_layout->addWidget(new QLabel(tr("Bit Index")));
  src_layout->addWidget(bit_idx_sb);
  src_layout->addStretch(0);

  QHBoxLayout *find_layout = new QHBoxLayout();
  find_layout->addWidget(new QLabel(tr("Bus")));
  find_layout->addWidget(find_bus_combo);
  find_layout->addWidget(new QLabel(tr("Equal")));
  equal_combo = new QComboBox(this);
  equal_combo->addItems({"Yes", "No"});
  find_layout->addWidget(equal_combo);
  min_msgs = new QLineEdit(this);
  min_msgs->setValidator(new QIntValidator(this));
  min_msgs->setText("100");
  find_layout->addWidget(new QLabel(tr("Min msg count")));
  find_layout->addWidget(min_msgs);
  search_btn = new QPushButton(tr("&Find"), this);
  find_layout->addWidget(search_btn);
  find_layout->addStretch(0);

  QGridLayout *grid_layout = new QGridLayout();
  grid_layout->addWidget(new QLabel("Find From:"), 0, 0);
  grid_layout->addLayout(src_layout, 0, 1);
  grid_layout->addWidget(new QLabel("Find In:"), 1, 0);
  grid_layout->addLayout(find_layout, 1, 1);
  main_layout->addLayout(grid_layout);

  table = new QTableWidget(this);
  table->setSelectionBehavior(QAbstractItemView::SelectRows);
  table->setSelectionMode(QAbstractItemView::SingleSelection);
  table->setEditTriggers(QAbstractItemView::NoEditTriggers);
  table->horizontalHeader()->setStretchLastSection(true);
  main_layout->addWidget(table);

  setMinimumSize({700, 500});
  QObject::connect(search_btn, &QPushButton::clicked, this, &FindSimilarBitsDlg::find);
  QObject::connect(table, &QTableWidget::doubleClicked, [this](const QModelIndex &index) {
    if (index.isValid()) {
      MessageId msg_id = {.source = (uint8_t)find_bus_combo->currentData().toUInt(), .address = table->item(index.row(), 0)->text().toUInt(0, 16)};
      emit openMessage(msg_id);
    }
  });
}

void FindSimilarBitsDlg::find() {
  search_btn->setEnabled(false);
  table->clear();
  uint32_t selected_address = msg_cb->currentData().toUInt();
  auto msg_mismatched = calcBits(src_bus_combo->currentText().toUInt(), selected_address, byte_idx_sb->value(), bit_idx_sb->value(),
                                 find_bus_combo->currentText().toUInt(), equal_combo->currentIndex() == 0, min_msgs->text().toInt());
  table->setRowCount(msg_mismatched.size());
  table->setColumnCount(6);
  table->setHorizontalHeaderLabels({"address", "byte idx", "bit idx", "mismatches", "total msgs", "% mismatched"});
  for (int i = 0; i < msg_mismatched.size(); ++i) {
    auto &m = msg_mismatched[i];
    table->setItem(i, 0, new QTableWidgetItem(QString("%1").arg(m.address, 1, 16)));
    table->setItem(i, 1, new QTableWidgetItem(QString::number(m.byte_idx)));
    table->setItem(i, 2, new QTableWidgetItem(QString::number(m.bit_idx)));
    table->setItem(i, 3, new QTableWidgetItem(QString::number(m.mismatches)));
    table->setItem(i, 4, new QTableWidgetItem(QString::number(m.total)));
    table->setItem(i, 5, new QTableWidgetItem(QString::number(m.perc, 'f', 2)));
  }
  search_btn->setEnabled(true);
}

QList<FindSimilarBitsDlg::mismatched_struct> FindSimilarBitsDlg::calcBits(uint8_t bus, uint32_t selected_address, int byte_idx,
                                                                          int bit_idx, uint8_t find_bus, bool equal, int min_msgs_cnt) {
  QHash<uint32_t, QVector<uint32_t>> mismatches;
  QHash<uint32_t, uint32_t> msg_count;
  auto events = can->rawEvents();
  int bit_to_find = -1;
  for (auto e : *events) {
    if (e->which == cereal::Event::Which::CAN) {
      for (const auto &c : e->event.getCan()) {
        uint8_t src = c.getSrc();
        uint32_t address = c.getAddress();
        const auto dat = c.getDat();
        if (src == bus) {
          if (address == selected_address && dat.size() > byte_idx) {
            bit_to_find = ((dat[byte_idx] >> (7 - bit_idx)) & 1) != 0;
          }
        }
        if (src == find_bus) {
          ++msg_count[address];
          if (bit_to_find == -1) continue;

          auto &mismatched = mismatches[address];
          if (mismatched.size() < dat.size() * 8) {
            mismatched.resize(dat.size() * 8);
          }
          for (int i = 0; i < dat.size(); ++i) {
            for (int j = 0; j < 8; ++j) {
              int bit = ((dat[i] >> (7 - j)) & 1) != 0;
              mismatched[i * 8 + j] += equal ? (bit != bit_to_find) : (bit == bit_to_find);
            }
          }
        }
      }
    }
  }

  QList<mismatched_struct> result;
  result.reserve(mismatches.size());
  for (auto it = mismatches.begin(); it != mismatches.end(); ++it) {
    if (auto cnt = msg_count[it.key()]; cnt > min_msgs_cnt) {
      auto &mismatched = it.value();
      for (int i = 0; i < mismatched.size(); ++i) {
        if (float perc = (mismatched[i] / (double)cnt) * 100; perc < 50) {
          result.push_back({it.key(), (uint32_t)i / 8, (uint32_t)i % 8, mismatched[i], cnt, perc});
        }
      }
    }
  }
  std::sort(result.begin(), result.end(), [](auto &l, auto &r) { return l.perc < r.perc; });
  return result;
}
