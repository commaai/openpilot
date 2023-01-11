#pragma once

#include <QComboBox>
#include <QDialog>
#include <QLineEdit>
#include <QTableWidget>

class FindSimilarBitsDlg : public QDialog {
public:
  FindSimilarBitsDlg(QWidget *parent);

private:
  struct mismatched_struct {
    uint32_t address, byte_idx, bit_idx, mismatches, total, perc;
  };
  QList<mismatched_struct> calcBits(uint8_t bus, int bit_to_find, int min_msgs_cnt);
  void find();

  QTableWidget *table;
  QComboBox *bus_combo, *bit_combo;
  QPushButton *search_btn;
  QLineEdit *min_msgs;
};
