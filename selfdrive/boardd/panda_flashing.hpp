#include <string>

#include "panda.h"

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

std::string get_firmware_fn();
// Ensures panda is running firmware or throws an exception if that is not an option
void update_panda(std::string serial="", std::string dfu_serial="");
