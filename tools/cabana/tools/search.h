#pragma once

#include <QDialog>
#include <QPushButton>
#include <QButtonGroup>
#include <QVBoxLayout>
#include <QLineEdit>
#include <QDoubleValidator>
#include <QComboBox>
#include <QSpinBox>


class SearchDlg : public QDialog {
  Q_OBJECT

public:
  SearchDlg(QWidget *parent);

private:
  void firstScan();
  void nextScan();
  void undoScan();

  uint32_t scan_bits_range_min = 1;
  uint32_t scan_bits_range_max = 32;

  uint64_t scan_value = 0;
};