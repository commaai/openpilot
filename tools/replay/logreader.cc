#include "tools/replay/logreader.h"

#include <algorithm>
#include <utility>
#include "tools/replay/filereader.h"
#include "tools/replay/util.h"
#include "common/util.h"

bool LogReader::load(const std::string &url, std::atomic<bool> *abort, bool local_cache, int chunk_size, int retries) {
  std::string data = FileReader(local_cache, chunk_size, retries).read(url, abort);
  if (!data.empty()) {
    if (url.find(".bz2") != std::string::npos || util::starts_with(data, "BZh9")) {
      data = decompressBZ2(data, abort);
    } else if (url.find(".zst") != std::string::npos || util::starts_with(data, "\x28\xB5\x2F\xFD")) {
      data = decompressZST(data, abort);
    }
  }

  bool success = !data.empty() && load(data.data(), data.size(), abort);
  if (filters_.empty())
    raw_ = std::move(data);
  return success;
}

bool LogReader::load(const char *data, size_t size, std::atomic<bool> *abort) {
  try {
    events.reserve(65000);
    kj::ArrayPtr<const capnp::word> words((const capnp::word *)data, size / sizeof(capnp::word));
    while (words.size() > 0 && !(abort && *abort)) {
      capnp::FlatArrayMessageReader reader(words);
      auto event = reader.getRoot<cereal::Event>();
      auto which = event.which();
      auto event_data = kj::arrayPtr(words.begin(), reader.getEnd());
      words = kj::arrayPtr(reader.getEnd(), words.end());

      if (!filters_.empty()) {
        if (which >= filters_.size() || !filters_[which])
          continue;
        auto buf = buffer_.allocate(event_data.size() * sizeof(capnp::word));
        memcpy(buf, event_data.begin(), event_data.size() * sizeof(capnp::word));
        event_data = kj::arrayPtr((const capnp::word *)buf, event_data.size());
      }

      uint64_t mono_time = event.getLogMonoTime();
      const Event &evt = events.emplace_back(which, mono_time, event_data);
      // Add encodeIdx packet again as a frame packet for the video stream
      if (evt.which == cereal::Event::ROAD_ENCODE_IDX ||
          evt.which == cereal::Event::DRIVER_ENCODE_IDX ||
          evt.which == cereal::Event::WIDE_ROAD_ENCODE_IDX) {
        auto idx = capnp::AnyStruct::Reader(event).getPointerSection()[0].getAs<cereal::EncodeIndex>();
        if (idx.getType() == cereal::EncodeIndex::Type::FULL_H_E_V_C) {
          uint64_t sof = idx.getTimestampSof();
          events.emplace_back(which, sof ? sof : mono_time, event_data, idx.getSegmentNum());
        }
      }
    }
  } catch (const kj::Exception &e) {
    rWarning("Failed to parse log : %s.\nRetrieved %zu events from corrupt log", e.getDescription().cStr(), events.size());
  }

  if (requires_migration) {
    migrateOldEvents();
  }

  if (!events.empty() && !(abort && *abort)) {
    events.shrink_to_fit();
    std::sort(events.begin(), events.end());
    return true;
  }
  return false;
}

void LogReader::migrateOldEvents() {
  size_t events_size = events.size();
  for (int i = 0; i < events_size; ++i) {
    // Check if the event is of the old CONTROLS_STATE type
    auto &event = events[i];
    if (event.which == cereal::Event::CONTROLS_STATE) {
      // Read the old event data
      capnp::FlatArrayMessageReader reader(event.data);
      auto old_evt = reader.getRoot<cereal::Event>();
      auto old_state = old_evt.getControlsState();

      // Migrate relevant fields from old CONTROLS_STATE to new SelfdriveState
      MessageBuilder msg;
      auto new_evt = msg.initEvent(old_evt.getValid());
      new_evt.setLogMonoTime(old_evt.getLogMonoTime());
      auto new_state = new_evt.initSelfdriveState();

      new_state.setActive(old_state.getActiveDEPRECATED());
      new_state.setAlertSize(old_state.getAlertSizeDEPRECATED());
      new_state.setAlertSound(old_state.getAlertSound2DEPRECATED());
      new_state.setAlertStatus(old_state.getAlertStatusDEPRECATED());
      new_state.setAlertText1(old_state.getAlertText1DEPRECATED());
      new_state.setAlertText2(old_state.getAlertText2DEPRECATED());
      new_state.setAlertType(old_state.getAlertTypeDEPRECATED());
      new_state.setEnabled(old_state.getEnabledDEPRECATED());
      new_state.setEngageable(old_state.getEngageableDEPRECATED());
      new_state.setExperimentalMode(old_state.getExperimentalModeDEPRECATED());
      new_state.setPersonality(old_state.getPersonalityDEPRECATED());
      new_state.setState(old_state.getStateDEPRECATED());

      // Serialize the new event to the buffer
      auto buf_size = msg.getSerializedSize();
      auto buf = buffer_.allocate(buf_size);
      msg.serializeToBuffer(reinterpret_cast<unsigned char *>(buf), buf_size);

      // Store the migrated event in the events list
      auto event_data = kj::arrayPtr(reinterpret_cast<const capnp::word *>(buf), buf_size);
      events.emplace_back(new_evt.which(), new_evt.getLogMonoTime(), event_data);
    } else if (event.which == cereal::Event::ONROAD_EVENTS_D_E_P_R_E_C_A_T_E_D) {
      // Read the old event data
      capnp::FlatArrayMessageReader reader(event.data);
      auto old_evt = reader.getRoot<cereal::Event>();
      auto old_state = old_evt.getOnroadEventsDEPRECATED();

      MessageBuilder msg;
      auto new_evt = msg.initEvent(old_evt.getValid());
      new_evt.setLogMonoTime(old_evt.getLogMonoTime());
      size_t new_onroad_events_size = old_state.size();
      auto new_onroad_events = new_evt.initOnroadEvents(new_onroad_events_size);
      for (size_t j = 0; j < new_onroad_events_size; j++) {
        cereal::OnroadEventDEPRECATED::Reader old_event = old_state[j];
        cereal::OnroadEvent::Builder new_event = new_onroad_events[j];

        switch (old_event.getName()) {
          case cereal::OnroadEventDEPRECATED::EventName::CAN_ERROR:
            new_event.setName(cereal::OnroadEvent::EventName::CAN_ERROR);
            break;
          case cereal::OnroadEventDEPRECATED::EventName::STEER_UNAVAILABLE:
            new_event.setName(cereal::OnroadEvent::EventName::STEER_UNAVAILABLE);
            break;
          case cereal::OnroadEventDEPRECATED::EventName::WRONG_GEAR:
            new_event.setName(cereal::OnroadEvent::EventName::WRONG_GEAR);
            break;
          case cereal::OnroadEventDEPRECATED::EventName::DOOR_OPEN:
            new_event.setName(cereal::OnroadEvent::EventName::DOOR_OPEN);
            break;
          case cereal::OnroadEventDEPRECATED::EventName::SEATBELT_NOT_LATCHED:
            new_event.setName(cereal::OnroadEvent::EventName::SEATBELT_NOT_LATCHED);
            break;
          case cereal::OnroadEventDEPRECATED::EventName::ESP_DISABLED:
            new_event.setName(cereal::OnroadEvent::EventName::ESP_DISABLED);
            break;
          case cereal::OnroadEventDEPRECATED::EventName::WRONG_CAR_MODE:
            new_event.setName(cereal::OnroadEvent::EventName::WRONG_CAR_MODE);
            break;
          case cereal::OnroadEventDEPRECATED::EventName::STEER_TEMP_UNAVAILABLE:
            new_event.setName(cereal::OnroadEvent::EventName::STEER_TEMP_UNAVAILABLE);
            break;
          case cereal::OnroadEventDEPRECATED::EventName::REVERSE_GEAR:
            new_event.setName(cereal::OnroadEvent::EventName::REVERSE_GEAR);
            break;
          case cereal::OnroadEventDEPRECATED::EventName::BUTTON_CANCEL:
            new_event.setName(cereal::OnroadEvent::EventName::BUTTON_CANCEL);
            break;
          case cereal::OnroadEventDEPRECATED::EventName::BUTTON_ENABLE:
            new_event.setName(cereal::OnroadEvent::EventName::BUTTON_ENABLE);
            break;
          case cereal::OnroadEventDEPRECATED::EventName::PEDAL_PRESSED:
            new_event.setName(cereal::OnroadEvent::EventName::PEDAL_PRESSED);
            break;
          case cereal::OnroadEventDEPRECATED::EventName::CRUISE_DISABLED:
            new_event.setName(cereal::OnroadEvent::EventName::CRUISE_DISABLED);
            break;
          case cereal::OnroadEventDEPRECATED::EventName::SPEED_TOO_LOW:
            new_event.setName(cereal::OnroadEvent::EventName::SPEED_TOO_LOW);
            break;
          case cereal::OnroadEventDEPRECATED::EventName::OUT_OF_SPACE:
            new_event.setName(cereal::OnroadEvent::EventName::OUT_OF_SPACE);
            break;
          case cereal::OnroadEventDEPRECATED::EventName::OVERHEAT:
            new_event.setName(cereal::OnroadEvent::EventName::OVERHEAT);
            break;
          case cereal::OnroadEventDEPRECATED::EventName::CALIBRATION_INCOMPLETE:
            new_event.setName(cereal::OnroadEvent::EventName::CALIBRATION_INCOMPLETE);
            break;
          case cereal::OnroadEventDEPRECATED::EventName::CALIBRATION_INVALID:
            new_event.setName(cereal::OnroadEvent::EventName::CALIBRATION_INVALID);
            break;
          case cereal::OnroadEventDEPRECATED::EventName::CONTROLS_MISMATCH:
            new_event.setName(cereal::OnroadEvent::EventName::CONTROLS_MISMATCH);
            break;
          case cereal::OnroadEventDEPRECATED::EventName::PCM_ENABLE:
            new_event.setName(cereal::OnroadEvent::EventName::PCM_ENABLE);
            break;
          case cereal::OnroadEventDEPRECATED::EventName::PCM_DISABLE:
            new_event.setName(cereal::OnroadEvent::EventName::PCM_DISABLE);
            break;
          case cereal::OnroadEventDEPRECATED::EventName::RADAR_FAULT:
            new_event.setName(cereal::OnroadEvent::EventName::RADAR_FAULT);
            break;
          case cereal::OnroadEventDEPRECATED::EventName::BRAKE_HOLD:
            new_event.setName(cereal::OnroadEvent::EventName::BRAKE_HOLD);
            break;
          case cereal::OnroadEventDEPRECATED::EventName::PARK_BRAKE:
            new_event.setName(cereal::OnroadEvent::EventName::PARK_BRAKE);
            break;
          case cereal::OnroadEventDEPRECATED::EventName::MANUAL_RESTART:
            new_event.setName(cereal::OnroadEvent::EventName::MANUAL_RESTART);
            break;
          case cereal::OnroadEventDEPRECATED::EventName::JOYSTICK_DEBUG:
            new_event.setName(cereal::OnroadEvent::EventName::JOYSTICK_DEBUG);
            break;
          case cereal::OnroadEventDEPRECATED::EventName::STEER_TEMP_UNAVAILABLE_SILENT:
            new_event.setName(cereal::OnroadEvent::EventName::STEER_TEMP_UNAVAILABLE_SILENT);
            break;
          case cereal::OnroadEventDEPRECATED::EventName::RESUME_REQUIRED:
            new_event.setName(cereal::OnroadEvent::EventName::RESUME_REQUIRED);
            break;
          case cereal::OnroadEventDEPRECATED::EventName::PRE_DRIVER_DISTRACTED:
            new_event.setName(cereal::OnroadEvent::EventName::PRE_DRIVER_DISTRACTED);
            break;
          case cereal::OnroadEventDEPRECATED::EventName::PROMPT_DRIVER_DISTRACTED:
            new_event.setName(cereal::OnroadEvent::EventName::PROMPT_DRIVER_DISTRACTED);
            break;
          case cereal::OnroadEventDEPRECATED::EventName::DRIVER_DISTRACTED:
            new_event.setName(cereal::OnroadEvent::EventName::DRIVER_DISTRACTED);
            break;
          case cereal::OnroadEventDEPRECATED::EventName::PRE_DRIVER_UNRESPONSIVE:
            new_event.setName(cereal::OnroadEvent::EventName::PRE_DRIVER_UNRESPONSIVE);
            break;
          case cereal::OnroadEventDEPRECATED::EventName::PROMPT_DRIVER_UNRESPONSIVE:
            new_event.setName(cereal::OnroadEvent::EventName::PROMPT_DRIVER_UNRESPONSIVE);
            break;
          case cereal::OnroadEventDEPRECATED::EventName::DRIVER_UNRESPONSIVE:
            new_event.setName(cereal::OnroadEvent::EventName::DRIVER_UNRESPONSIVE);
            break;
          case cereal::OnroadEventDEPRECATED::EventName::BELOW_STEER_SPEED:
            new_event.setName(cereal::OnroadEvent::EventName::BELOW_STEER_SPEED);
            break;
          case cereal::OnroadEventDEPRECATED::EventName::LOW_BATTERY:
            new_event.setName(cereal::OnroadEvent::EventName::LOW_BATTERY);
            break;
          case cereal::OnroadEventDEPRECATED::EventName::PARAMSD_TEMPORARY_ERROR:
            new_event.setName(cereal::OnroadEvent::EventName::PARAMSD_TEMPORARY_ERROR);
            break;
          case cereal::OnroadEventDEPRECATED::EventName::ACC_FAULTED:
            new_event.setName(cereal::OnroadEvent::EventName::ACC_FAULTED);
            break;
          case cereal::OnroadEventDEPRECATED::EventName::SENSOR_DATA_INVALID:
            new_event.setName(cereal::OnroadEvent::EventName::SENSOR_DATA_INVALID);
            break;
          case cereal::OnroadEventDEPRECATED::EventName::COMM_ISSUE:
            new_event.setName(cereal::OnroadEvent::EventName::COMM_ISSUE);
            break;
          case cereal::OnroadEventDEPRECATED::EventName::TOO_DISTRACTED:
            new_event.setName(cereal::OnroadEvent::EventName::TOO_DISTRACTED);
            break;
          case cereal::OnroadEventDEPRECATED::EventName::POSENET_INVALID:
            new_event.setName(cereal::OnroadEvent::EventName::POSENET_INVALID);
            break;
          case cereal::OnroadEventDEPRECATED::EventName::PRE_LANE_CHANGE_LEFT:
            new_event.setName(cereal::OnroadEvent::EventName::PRE_LANE_CHANGE_LEFT);
            break;
          case cereal::OnroadEventDEPRECATED::EventName::PRE_LANE_CHANGE_RIGHT:
            new_event.setName(cereal::OnroadEvent::EventName::PRE_LANE_CHANGE_RIGHT);
            break;
          case cereal::OnroadEventDEPRECATED::EventName::LANE_CHANGE:
            new_event.setName(cereal::OnroadEvent::EventName::LANE_CHANGE);
            break;
          case cereal::OnroadEventDEPRECATED::EventName::LOW_MEMORY:
            new_event.setName(cereal::OnroadEvent::EventName::LOW_MEMORY);
            break;
          case cereal::OnroadEventDEPRECATED::EventName::STOCK_AEB:
            new_event.setName(cereal::OnroadEvent::EventName::STOCK_AEB);
            break;
          case cereal::OnroadEventDEPRECATED::EventName::LDW:
            new_event.setName(cereal::OnroadEvent::EventName::LDW);
            break;
          case cereal::OnroadEventDEPRECATED::EventName::CAR_UNRECOGNIZED:
            new_event.setName(cereal::OnroadEvent::EventName::CAR_UNRECOGNIZED);
            break;
          case cereal::OnroadEventDEPRECATED::EventName::INVALID_LKAS_SETTING:
            new_event.setName(cereal::OnroadEvent::EventName::INVALID_LKAS_SETTING);
            break;
          case cereal::OnroadEventDEPRECATED::EventName::SPEED_TOO_HIGH:
            new_event.setName(cereal::OnroadEvent::EventName::SPEED_TOO_HIGH);
            break;
          case cereal::OnroadEventDEPRECATED::EventName::LANE_CHANGE_BLOCKED:
            new_event.setName(cereal::OnroadEvent::EventName::LANE_CHANGE_BLOCKED);
            break;
          case cereal::OnroadEventDEPRECATED::EventName::RELAY_MALFUNCTION:
            new_event.setName(cereal::OnroadEvent::EventName::RELAY_MALFUNCTION);
            break;
          case cereal::OnroadEventDEPRECATED::EventName::PRE_ENABLE_STANDSTILL:
            new_event.setName(cereal::OnroadEvent::EventName::PRE_ENABLE_STANDSTILL);
            break;
          case cereal::OnroadEventDEPRECATED::EventName::STOCK_FCW:
            new_event.setName(cereal::OnroadEvent::EventName::STOCK_FCW);
            break;
          case cereal::OnroadEventDEPRECATED::EventName::STARTUP:
            new_event.setName(cereal::OnroadEvent::EventName::STARTUP);
            break;
          case cereal::OnroadEventDEPRECATED::EventName::STARTUP_NO_CAR:
            new_event.setName(cereal::OnroadEvent::EventName::STARTUP_NO_CAR);
            break;
          case cereal::OnroadEventDEPRECATED::EventName::STARTUP_NO_CONTROL:
            new_event.setName(cereal::OnroadEvent::EventName::STARTUP_NO_CONTROL);
            break;
          case cereal::OnroadEventDEPRECATED::EventName::STARTUP_MASTER:
            new_event.setName(cereal::OnroadEvent::EventName::STARTUP_MASTER);
            break;
          case cereal::OnroadEventDEPRECATED::EventName::FCW:
            new_event.setName(cereal::OnroadEvent::EventName::FCW);
            break;
          case cereal::OnroadEventDEPRECATED::EventName::STEER_SATURATED:
            new_event.setName(cereal::OnroadEvent::EventName::STEER_SATURATED);
            break;
          case cereal::OnroadEventDEPRECATED::EventName::BELOW_ENGAGE_SPEED:
            new_event.setName(cereal::OnroadEvent::EventName::BELOW_ENGAGE_SPEED);
            break;
          case cereal::OnroadEventDEPRECATED::EventName::NO_GPS:
            new_event.setName(cereal::OnroadEvent::EventName::NO_GPS);
            break;
          case cereal::OnroadEventDEPRECATED::EventName::WRONG_CRUISE_MODE:
            new_event.setName(cereal::OnroadEvent::EventName::WRONG_CRUISE_MODE);
            break;
          case cereal::OnroadEventDEPRECATED::EventName::MODELD_LAGGING:
            new_event.setName(cereal::OnroadEvent::EventName::MODELD_LAGGING);
            break;
          case cereal::OnroadEventDEPRECATED::EventName::DEVICE_FALLING:
            new_event.setName(cereal::OnroadEvent::EventName::DEVICE_FALLING);
            break;
          case cereal::OnroadEventDEPRECATED::EventName::FAN_MALFUNCTION:
            new_event.setName(cereal::OnroadEvent::EventName::FAN_MALFUNCTION);
            break;
          case cereal::OnroadEventDEPRECATED::EventName::CAMERA_MALFUNCTION:
            new_event.setName(cereal::OnroadEvent::EventName::CAMERA_MALFUNCTION);
            break;
          case cereal::OnroadEventDEPRECATED::EventName::PROCESS_NOT_RUNNING:
            new_event.setName(cereal::OnroadEvent::EventName::PROCESS_NOT_RUNNING);
            break;
          case cereal::OnroadEventDEPRECATED::EventName::DASHCAM_MODE:
            new_event.setName(cereal::OnroadEvent::EventName::DASHCAM_MODE);
            break;
          case cereal::OnroadEventDEPRECATED::EventName::SELFDRIVE_INITIALIZING:
            new_event.setName(cereal::OnroadEvent::EventName::SELFDRIVE_INITIALIZING);
            break;
          case cereal::OnroadEventDEPRECATED::EventName::USB_ERROR:
            new_event.setName(cereal::OnroadEvent::EventName::USB_ERROR);
            break;
          case cereal::OnroadEventDEPRECATED::EventName::LOCATIOND_TEMPORARY_ERROR:
            new_event.setName(cereal::OnroadEvent::EventName::LOCATIOND_TEMPORARY_ERROR);
            break;
          case cereal::OnroadEventDEPRECATED::EventName::CRUISE_MISMATCH:
            new_event.setName(cereal::OnroadEvent::EventName::CRUISE_MISMATCH);
            break;
          case cereal::OnroadEventDEPRECATED::EventName::GAS_PRESSED_OVERRIDE:
            new_event.setName(cereal::OnroadEvent::EventName::GAS_PRESSED_OVERRIDE);
            break;
          case cereal::OnroadEventDEPRECATED::EventName::COMM_ISSUE_AVG_FREQ:
            new_event.setName(cereal::OnroadEvent::EventName::COMM_ISSUE_AVG_FREQ);
            break;
          case cereal::OnroadEventDEPRECATED::EventName::CAMERA_FRAME_RATE:
            new_event.setName(cereal::OnroadEvent::EventName::CAMERA_FRAME_RATE);
            break;
          case cereal::OnroadEventDEPRECATED::EventName::CAN_BUS_MISSING:
            new_event.setName(cereal::OnroadEvent::EventName::CAN_BUS_MISSING);
            break;
          case cereal::OnroadEventDEPRECATED::EventName::SELFDRIVED_LAGGING:
            new_event.setName(cereal::OnroadEvent::EventName::SELFDRIVED_LAGGING);
            break;
          case cereal::OnroadEventDEPRECATED::EventName::RESUME_BLOCKED:
            new_event.setName(cereal::OnroadEvent::EventName::RESUME_BLOCKED);
            break;
          case cereal::OnroadEventDEPRECATED::EventName::STEER_OVERRIDE:
            new_event.setName(cereal::OnroadEvent::EventName::STEER_OVERRIDE);
            break;
          case cereal::OnroadEventDEPRECATED::EventName::STEER_TIME_LIMIT:
            new_event.setName(cereal::OnroadEvent::EventName::STEER_TIME_LIMIT);
            break;
          case cereal::OnroadEventDEPRECATED::EventName::VEHICLE_SENSORS_INVALID:
            new_event.setName(cereal::OnroadEvent::EventName::VEHICLE_SENSORS_INVALID);
            break;
          case cereal::OnroadEventDEPRECATED::EventName::CALIBRATION_RECALIBRATING:
            new_event.setName(cereal::OnroadEvent::EventName::CALIBRATION_RECALIBRATING);
            break;
          case cereal::OnroadEventDEPRECATED::EventName::LOCATIOND_PERMANENT_ERROR:
            new_event.setName(cereal::OnroadEvent::EventName::LOCATIOND_PERMANENT_ERROR);
            break;
          case cereal::OnroadEventDEPRECATED::EventName::PARAMSD_PERMANENT_ERROR:
            new_event.setName(cereal::OnroadEvent::EventName::PARAMSD_PERMANENT_ERROR);
            break;
          case cereal::OnroadEventDEPRECATED::EventName::ACTUATORS_API_UNAVAILABLE:
            new_event.setName(cereal::OnroadEvent::EventName::ACTUATORS_API_UNAVAILABLE);
            break;
          case cereal::OnroadEventDEPRECATED::EventName::ESP_ACTIVE:
            new_event.setName(cereal::OnroadEvent::EventName::ESP_ACTIVE);
            break;
          case cereal::OnroadEventDEPRECATED::EventName::PERSONALITY_CHANGED:
            new_event.setName(cereal::OnroadEvent::EventName::PERSONALITY_CHANGED);
            break;
          case cereal::OnroadEventDEPRECATED::EventName::AEB:
            new_event.setName(cereal::OnroadEvent::EventName::AEB);
            break;
          case cereal::OnroadEventDEPRECATED::EventName::LONGITUDINAL_MANEUVER:
            new_event.setName(cereal::OnroadEvent::EventName::LONGITUDINAL_MANEUVER);
            break;
          case cereal::OnroadEventDEPRECATED::EventName::STARTUP_NO_SEC_OC_KEY:
            new_event.setName(cereal::OnroadEvent::EventName::STARTUP_NO_SEC_OC_KEY);
            break;
          default:
            break;
        }

        new_event.setEnable(old_event.getEnable());
        new_event.setNoEntry(old_event.getNoEntry());
        new_event.setWarning(old_event.getWarning());
        new_event.setUserDisable(old_event.getUserDisable());
        new_event.setSoftDisable(old_event.getSoftDisable());
        new_event.setImmediateDisable(old_event.getImmediateDisable());
        new_event.setPreEnable(old_event.getPreEnable());
        new_event.setPermanent(old_event.getPermanent());
        new_event.setOverrideLateral(old_event.getOverrideLateral());
        new_event.setOverrideLongitudinal(old_event.getOverrideLongitudinal());
      }

      // Serialize the new event to the buffer
      auto buf_size = msg.getSerializedSize();
      auto buf = buffer_.allocate(buf_size);
      msg.serializeToBuffer(reinterpret_cast<unsigned char *>(buf), buf_size);

      // Store the migrated event in the events list
      auto event_data = kj::arrayPtr(reinterpret_cast<const capnp::word *>(buf), buf_size);
      events.emplace_back(new_evt.which(), new_evt.getLogMonoTime(), event_data);
    }
  }
}
