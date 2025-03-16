#include "selfdrive/pandad/pandad.h"
#include "cereal/messaging/messaging.h"
#include "common/swaglog.h"

void PandaSafety::configureSafetyMode() {
  bool is_onroad = params_.getBool("IsOnroad");

  if (is_onroad && !safety_configured_) {
    updateMultiplexingMode();

    auto car_params = fetchCarParams();
    if (!car_params.empty()) {
      LOGW("got %lu bytes CarParams", car_params[0].size());
      LOGW("got %lu bytes CarParamsSP", car_params[1].size());
      setSafetyMode(car_params);
      safety_configured_ = true;
    }
  } else if (!is_onroad) {
    initialized_ = false;
    safety_configured_ = false;
    log_once_ = false;
  }
}

void PandaSafety::updateMultiplexingMode() {
  // Initialize to ELM327 without OBD multiplexing for initial fingerprinting
  if (!initialized_) {
    prev_obd_multiplexing_ = false;
    for (int i = 0; i < pandas_.size(); ++i) {
      pandas_[i]->set_safety_model(cereal::CarParams::SafetyModel::ELM327, 1U);
    }
    initialized_ = true;
  }

  // Switch between multiplexing modes based on the OBD multiplexing request
  bool obd_multiplexing_requested = params_.getBool("ObdMultiplexingEnabled");
  if (obd_multiplexing_requested != prev_obd_multiplexing_) {
    for (int i = 0; i < pandas_.size(); ++i) {
      const uint16_t safety_param = (i > 0 || !obd_multiplexing_requested) ? 1U : 0U;
      pandas_[i]->set_safety_model(cereal::CarParams::SafetyModel::ELM327, safety_param);
    }
    prev_obd_multiplexing_ = obd_multiplexing_requested;
    params_.putBool("ObdMultiplexingChanged", true);
  }
}

// TODO-SP: Use structs instead of vector
std::vector<std::string> PandaSafety::fetchCarParams() {
  if (!params_.getBool("FirmwareQueryDone")) {
    return {};
  }

  if (!log_once_) {
    LOGW("Finished FW query, Waiting for params to set safety model");
    log_once_ = true;
  }

  if (!params_.getBool("ControlsReady")) {
    return {};
  }
  return {params_.get("CarParams"), params_.get("CarParamsSP")};
}

// TODO-SP: Use structs instead of vector
void PandaSafety::setSafetyMode(const std::vector<std::string> &params_string) {
  AlignedBuffer aligned_buf;
  AlignedBuffer aligned_buf_sp;

  capnp::FlatArrayMessageReader cmsg(aligned_buf.align(params_string[0].data(), params_string[0].size()));
  cereal::CarParams::Reader car_params = cmsg.getRoot<cereal::CarParams>();

  capnp::FlatArrayMessageReader cmsg_sp(aligned_buf_sp.align(params_string[1].data(), params_string[1].size()));
  cereal::CarParamsSP::Reader car_params_sp = cmsg_sp.getRoot<cereal::CarParamsSP>();

  auto safety_configs = car_params.getSafetyConfigs();
  uint16_t alternative_experience = car_params.getAlternativeExperience();
  uint16_t safety_param_sp = car_params_sp.getSafetyParam();

  for (int i = 0; i < pandas_.size(); ++i) {
    // Default to SILENT safety model if not specified
    cereal::CarParams::SafetyModel safety_model = cereal::CarParams::SafetyModel::SILENT;
    uint16_t safety_param = 0U;
    if (i < safety_configs.size()) {
      safety_model = safety_configs[i].getSafetyModel();
      safety_param = safety_configs[i].getSafetyParam();
    }

    LOGW("Panda %d: setting safety model: %d, param: %d, alternative experience: %d, param_sp: %d", i, (int)safety_model, safety_param, alternative_experience, safety_param_sp);
    pandas_[i]->set_alternative_experience(alternative_experience, safety_param_sp);
    pandas_[i]->set_safety_model(safety_model, safety_param);
  }
}

bool PandaSafety::getOffroadMode() {
  auto offroad_mode = params_.getBool("OffroadMode");
  return offroad_mode;
}
