#include "tools/cabana/tools/findsimilarbits.h"

#include <QHeaderView>
#include <QHBoxLayout>
#include <QIntValidator>
#include <QLabel>
#include <QPushButton>
#include <QRadioButton>

#include "tools/cabana/canmessages.h"
#include "tools/cabana/dbcmanager.h"

FindSimilarBitsDlg::FindSimilarBitsDlg(QWidget *parent) : QDialog(parent) {
  setWindowTitle(tr("Find similar bits"));
  QVBoxLayout *main_layout = new QVBoxLayout(this);

  QHBoxLayout *form_layout = new QHBoxLayout();
  bus_combo = new QComboBox(this);
  QSet<uint8_t> bus_set;
  for (auto it = can->can_msgs.begin(); it != can->can_msgs.end(); ++it) {
    bus_set << DBCManager::parseId(it.key()).first;
  }
  for (uint8_t bus : bus_set) {
    bus_combo->addItem(QString::number(bus));
  }
  bus_combo->model()->sort(0);
  bus_combo->setCurrentIndex(0);
  form_layout->addWidget(new QLabel("Bus"));
  form_layout->addWidget(bus_combo);

  bit_combo = new QComboBox(this);
  bit_combo->addItems({"0", "1"});
  bit_combo->setCurrentIndex(1);
  form_layout->addWidget(new QLabel("Bit"));
  form_layout->addWidget(bit_combo);

  min_msgs = new QLineEdit(this);
  min_msgs->setValidator(new QIntValidator(this));
  min_msgs->setText("100");
  form_layout->addWidget(new QLabel("Min msg count"));
  form_layout->addWidget(min_msgs);
  search_btn = new QPushButton(tr("&Find"), this);
  form_layout->addWidget(search_btn);
  form_layout->addStretch(1);
  main_layout->addLayout(form_layout);

  table = new QTableWidget(this);
  table->setEditTriggers(QAbstractItemView::NoEditTriggers);
  table->horizontalHeader()->setStretchLastSection(true);
  main_layout->addWidget(table);

  setMinimumSize({700, 500});
  QObject::connect(search_btn, &QPushButton::clicked, this, &FindSimilarBitsDlg::find);
}

void FindSimilarBitsDlg::find() {
  search_btn->setEnabled(false);
  table->clear();
  auto msg_mismatched = calcBits(bus_combo->currentText().toUInt(), bit_combo->currentIndex(), min_msgs->text().toInt());
  table->setRowCount(msg_mismatched.size());
  table->setColumnCount(6);
  table->setHorizontalHeaderLabels({"address", "byte idx", "bit idx", "mismatches", "total", "perc%"});
  for (int i = 0; i < msg_mismatched.size(); ++i) {
    auto &m = msg_mismatched[i];
    table->setItem(i, 0, new QTableWidgetItem(QString("%1").arg(m.address, 1, 16)));
    table->setItem(i, 1, new QTableWidgetItem(QString::number(m.byte_idx)));
    table->setItem(i, 2, new QTableWidgetItem(QString::number(m.bit_idx)));
    table->setItem(i, 3, new QTableWidgetItem(QString::number(m.mismatches)));
    table->setItem(i, 4, new QTableWidgetItem(QString::number(m.total)));
    table->setItem(i, 5, new QTableWidgetItem(QString::number(m.perc)));
  }
  search_btn->setEnabled(true);
}

QList<FindSimilarBitsDlg::mismatched_struct> FindSimilarBitsDlg::calcBits(uint8_t bus, int bit_to_find, int min_msgs_cnt) {
  QHash<uint32_t, QVector<uint32_t>> mismatches;
  QHash<uint32_t, uint32_t> msg_count;
  auto events = can->events();
  for (auto e : *events) {
    if (e->which == cereal::Event::Which::CAN) {
      for (const auto &c : e->event.getCan()) {
        if (c.getSrc() == bus) {
          uint32_t address = c.getAddress();
          ++msg_count[address];
          auto &mismatched = mismatches[address];
          const auto dat = c.getDat();
          if (mismatched.size() < dat.size() * 8) {
            mismatched.resize(dat.size() * 8);
          }
          for (int i = 0; i < dat.size(); ++i) {
            for (int j = 0; j < 8; ++j) {
              int bit = ((dat[i] >> (7 - j)) & 1) != 0;
              mismatched[i * 8 + j] += (bit != bit_to_find);
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
        if (uint32_t perc = (mismatched[i] / (double)cnt) * 100; perc < 50) {
          result.push_back({it.key(), (uint32_t)i / 8, (uint32_t)i % 8, mismatched[i], cnt, perc});
        }
      }
    }
  }
  std::sort(result.begin(), result.end(), [](auto &l, auto &r) { return l.perc > r.perc; });
  return result;
}
