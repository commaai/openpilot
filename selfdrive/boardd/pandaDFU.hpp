#include <string>

#include "panda.h"
#include "panda_flashing.hpp"

class PandaDFU : public PandaComm{
private:
  void status();
  void clear_status();
  void erase(int adress);
  void program(int adress, std::string program);
  void reset();
  void program_bootstub(std::string program_file);
public:
  PandaDFU(std::string dfu_serial="");
  void recover();

};