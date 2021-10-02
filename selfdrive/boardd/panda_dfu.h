#include <string>

#include "panda.h"
#include "panda_flashing.h"

class PandaDFU : public PandaComm{
public:
  PandaDFU(std::string dfu_serial="");
  void recover();

private:
  void status();
  std::array<uint8_t, 6> get_status();
  void clear_status();
  void erase(int adress);
  void program(int adress, std::string program);
  void reset();
  void program_bootstub(std::string program_file);
};
