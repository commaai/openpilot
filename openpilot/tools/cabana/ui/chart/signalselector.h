#pragma once

#include <string>
#include <vector>

#include "imgui.h"
#include "tools/cabana/dbc/dbcmanager.h"

// QDialog::exec() is non-blocking here: open(), draw() every frame until it returns false, then check accepted()
class SignalSelector {
public:
  struct ListItem {
    ListItem(const MessageId &msg_id, const cabana::Signal *sig) : msg_id(msg_id), sig(sig) {}
    MessageId msg_id;
    const cabana::Signal *sig;
  };

  SignalSelector(std::string title);
  std::vector<ListItem *> seletedItems();
  inline void addSelected(const MessageId &id, const cabana::Signal *sig) { addItemToList(selected_list, id, sig, true); }
  void open() { open_ = true; show_ = false; accepted_ = false; }
  bool draw();  // false once the dialog is closed
  bool accepted() const { return accepted_; }

private:
  void updateAvailableList(int index);
  void addItemToList(std::vector<ListItem> &parent, const MessageId id, const cabana::Signal *sig, bool show_msg_name = false);
  void add(ListItem *item);
  void remove(ListItem *item);
  void drawList(const char *id, std::vector<ListItem> &list, int *current_row, bool show_msg_name, bool *double_clicked, const ImVec2 &size);

  struct ComboItem {
    std::string text;
    MessageId id;
  };
  std::string title_;
  std::vector<ComboItem> msgs_combo;
  int msgs_combo_index_ = -1;
  std::string msgs_combo_filter_;
  std::vector<ListItem> available_list;
  std::vector<ListItem> selected_list;
  int available_row_ = -1;
  int selected_row_ = -1;
  bool accepted_ = false;
  bool open_ = false;
  bool show_ = false;
};
