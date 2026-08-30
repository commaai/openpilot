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
  struct mismatched_struct {
    uint32_t address, byte_idx, bit_idx, mismatches, total;
    float perc;
  };
  std::vector<mismatched_struct> calcBits(uint8_t bus, uint32_t selected_address, int byte_idx, int bit_idx, uint8_t find_bus,
                                    bool equal, int min_msgs_cnt);
  void find();
  void drawTable();

  std::vector<mismatched_struct> table;  // rows, cleared by find()
  bool table_has_columns = false;
  std::vector<int> bus_items;            // src_bus_combo / find_bus_combo items
  int src_bus_combo = 0, find_bus_combo = 0;
  std::vector<std::pair<std::string, uint32_t>> msg_items;  // (name, address)
  int msg_cb = 0;
  int equal_combo = 0;
  int byte_idx_sb = 0, bit_idx_sb = 0;
  bool search_btn_enabled = true;
  std::string min_msgs = "100";
};
