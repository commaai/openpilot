#pragma once

#include <QComboBox>
#include <QDialog>
#include <QLineEdit>
#include <QSpinBox>
#include <QTableWidget>

class FindSimilarBitsDlg : public QDialog {
public:
  FindSimilarBitsDlg(QWidget *parent);

private:
  struct mismatched_struct {
    uint32_t address, byte_idx, bit_idx, mismatches, total;
    float perc;
  };
  QList<mismatched_struct> calcBits(uint8_t bus, uint32_t selected_address, int byte_idx, int bit_idx, int min_msgs_cnt);
  void find();

  QTableWidget *table;
  QComboBox *bus_combo, *msg_cb;
  QSpinBox *byte_idx_sb, *bit_idx_sb;
  QPushButton *search_btn;
  QLineEdit *min_msgs;
};
