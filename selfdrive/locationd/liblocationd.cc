#include "locationd.h"

extern "C" {
  typedef Localizer* Localizer_t;

  Localizer *localizer_init(bool has_ublox) {
    return new Localizer();
  }

  void localizer_get_message_bytes(Localizer *localizer, bool inputsOK, bool sensorsOK, bool gpsOK, bool msgValid,
                                   char *buff, size_t buff_size) {
    MessageBuilder msg_builder;
    kj::ArrayPtr<char> arr = localizer->get_message_bytes(msg_builder, inputsOK, sensorsOK, gpsOK, msgValid).asChars();
    assert(buff_size >= arr.size());
    memcpy(buff, arr.begin(), arr.size());
  }

  void localizer_handle_msg_bytes(Localizer *localizer, const char *data, size_t size) {
    localizer->handle_msg_bytes(data, size);
  }

  void get_filter_internals(Localizer *localizer, double *state_buff, double *std_buff){
    Eigen::VectorXd state = localizer->get_state();
    memcpy(state_buff, state.data(), sizeof(double) * state.size());
    Eigen::VectorXd stdev = localizer->get_stdev();
    memcpy(std_buff, stdev.data(), sizeof(double) * stdev.size());
  }

  bool is_gps_ok(Localizer *localizer){
    return localizer->is_gps_ok();
  }

  bool are_inputs_ok(Localizer *localizer){
    return localizer->are_inputs_ok();
  }

  void observation_timings_invalid_reset(Localizer *localizer){
    localizer->observation_timings_invalid_reset();
  }

}
