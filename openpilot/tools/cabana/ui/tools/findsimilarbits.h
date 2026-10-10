#pragma once

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "tools/cabana/dbc/dbcmanager.h"
#include "tools/cabana/ui/tools/tooldialog.h"

class FindSimilarBitsDlg : public ToolDialog {
public:
  FindSimilarBitsDlg();
  bool draw() override;

  Observable<const MessageId &> openMessage;

private:
  struct Mismatch {
    uint32_t address, byte_idx, bit_idx, mismatches, total;
    float perc;
  };
  std::vector<Mismatch> calcBits(uint8_t bus, uint32_t selected_address, int byte_idx, int bit_idx, uint8_t find_bus,
                                 bool equal, int min_msgs_cnt);
  uint8_t busAt(int index) const { return index < (int)bus_items_.size() ? bus_items_[index] : 0; }
  void updateMessages();
  void find();
  void drawTable();

  std::vector<Mismatch> table_;  // rows, replaced by find()
  bool table_has_columns_ = false;
  std::vector<int> bus_items_;   // the source and the find bus combos
  int src_bus_ = 0, find_bus_ = 0;
  std::vector<std::pair<std::string, uint32_t>> msg_items_;  // (name, address) of the source bus
  std::vector<std::string> msg_names_;
  int msg_index_ = 0;
  int equal_ = 0;
  int byte_idx_ = 0, bit_idx_ = 0;
  int min_msgs_ = 100;
};
