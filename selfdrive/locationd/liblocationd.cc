#include "locationd.h"

extern "C" {
  typedef Localizer* Localizer_t;

  Localizer *localizer_init() {
    return new Localizer();
  }

  void localizer_get_message_bytes(Localizer *localizer, uint64_t logMonoTime,
    bool inputsOK, bool sensorsOK, bool gpsOK, char *buff)
  {
    MessageBuilder msg_builder;
    kj::ArrayPtr<char> arr = localizer->get_message_bytes(msg_builder, logMonoTime, inputsOK, sensorsOK, gpsOK).asChars();
    memcpy(buff, arr.begin(), arr.size());
  }

  void localizer_handle_msg_bytes(Localizer *localizer, const char *data, size_t size) {
    localizer->handle_msg_bytes(data, size);
  }
}
