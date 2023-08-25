#pragma once

#include <QComboBox>
#include <QDialog>
#include <QLineEdit>
#include <QSpinBox>
#include <QTableWidget>

#include "tools/cabana/dbc/dbcmanager.h"

class FindSimilarBitsDlg : public QDialog {
  Q_OBJECT

public:
  FindSimilarBitsDlg(QWidget *parent);

signals:
  void openMessage(const MessageId &msg_id);

private:
  struct mismatched_struct {
    uint32_t address, byte_idx, bit_idx, mismatches, total;
    float perc;
  };
  QList<mismatched_struct> calcBits(uint8_t bus, uint32_t selected_address, int byte_idx, int bit_idx, uint8_t find_bus,
                                    bool equal, int min_msgs_cnt);
  void find();

  QTableWidget *table;
  QComboBox *src_bus_combo, *find_bus_combo, *msg_cb, *equal_combo;
  QSpinBox *byte_idx_sb, *bit_idx_sb;
  QPushButton *search_btn;
  QLineEdit *min_msgs;
};
