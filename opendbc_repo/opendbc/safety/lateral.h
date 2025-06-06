#include "opendbc/safety/safety_declarations.h"

// ISO 11270
static const float ISO_LATERAL_ACCEL = 3.0;  // m/s^2

static const float EARTH_G = 9.81;
static const float AVERAGE_ROAD_ROLL = 0.06;  // ~3.4 degrees, 6% superelevation

// check that commanded torque value isn't too far from measured
static bool dist_to_meas_check(int val, int val_last, struct sample_t *val_meas,
                        const int MAX_RATE_UP, const int MAX_RATE_DOWN, const int MAX_ERROR) {

  // *** val rate limit check ***
  int highest_allowed_rl = MAX(val_last, 0) + MAX_RATE_UP;
  int lowest_allowed_rl = MIN(val_last, 0) - MAX_RATE_UP;

  // if we've exceeded the meas val, we must start moving toward 0
  int highest_allowed = MIN(highest_allowed_rl, MAX(val_last - MAX_RATE_DOWN, MAX(val_meas->max, 0) + MAX_ERROR));
  int lowest_allowed = MAX(lowest_allowed_rl, MIN(val_last + MAX_RATE_DOWN, MIN(val_meas->min, 0) - MAX_ERROR));

  // check for violation
  return max_limit_check(val, highest_allowed, lowest_allowed);
}

// check that commanded value isn't fighting against driver
static bool driver_limit_check(int val, int val_last, const struct sample_t *val_driver,
                        const int MAX_VAL, const int MAX_RATE_UP, const int MAX_RATE_DOWN,
                        const int MAX_ALLOWANCE, const int DRIVER_FACTOR) {

  // torque delta/rate limits
  int highest_allowed_rl = MAX(val_last, 0) + MAX_RATE_UP;
  int lowest_allowed_rl = MIN(val_last, 0) - MAX_RATE_UP;

  // driver
  int driver_max_limit = MAX_VAL + (MAX_ALLOWANCE + val_driver->max) * DRIVER_FACTOR;
  int driver_min_limit = -MAX_VAL + (-MAX_ALLOWANCE + val_driver->min) * DRIVER_FACTOR;

  // if we've exceeded the applied torque, we must start moving toward 0
  int highest_allowed = MIN(highest_allowed_rl, MAX(val_last - MAX_RATE_DOWN,
                                             MAX(driver_max_limit, 0)));
  int lowest_allowed = MAX(lowest_allowed_rl, MIN(val_last + MAX_RATE_DOWN,
                                           MIN(driver_min_limit, 0)));

  // check for violation
  return max_limit_check(val, highest_allowed, lowest_allowed);
}

// real time check, mainly used for steer torque rate limiter
static bool rt_torque_rate_limit_check(int val, int val_last, const int MAX_RT_DELTA) {

  // *** torque real time rate limit check ***
  int highest_val = MAX(val_last, 0) + MAX_RT_DELTA;
  int lowest_val = MIN(val_last, 0) - MAX_RT_DELTA;

  // check for violation
  return max_limit_check(val, highest_val, lowest_val);
}

// Safety checks for torque-based steering commands
bool steer_torque_cmd_checks(int desired_torque, int steer_req, const TorqueSteeringLimits limits) {
  bool violation = false;
  uint32_t ts = microsecond_timer_get();

  if (controls_allowed) {
    // Some safety models support variable torque limit based on vehicle speed
    int max_torque = limits.max_torque;
    if (limits.dynamic_max_torque) {
      const float fudged_speed = (vehicle_speed.min / VEHICLE_SPEED_FACTOR) - 1.;
      max_torque = interpolate(limits.max_torque_lookup, fudged_speed) + 1;
      max_torque = CLAMP(max_torque, -limits.max_torque, limits.max_torque);
    }

    // *** global torque limit check ***
    violation |= max_limit_check(desired_torque, max_torque, -max_torque);

    // *** torque rate limit check ***
    if (limits.type == TorqueDriverLimited) {
      violation |= driver_limit_check(desired_torque, desired_torque_last, &torque_driver,
                                      max_torque, limits.max_rate_up, limits.max_rate_down,
                                      limits.driver_torque_allowance, limits.driver_torque_multiplier);
    } else {
      violation |= dist_to_meas_check(desired_torque, desired_torque_last, &torque_meas,
                                      limits.max_rate_up, limits.max_rate_down, limits.max_torque_error);
    }
    desired_torque_last = desired_torque;

    // *** torque real time rate limit check ***
    violation |= rt_torque_rate_limit_check(desired_torque, rt_torque_last, limits.max_rt_delta);

    // every RT_INTERVAL set the new limits
    uint32_t ts_elapsed = get_ts_elapsed(ts, ts_torque_check_last);
    if (ts_elapsed > MAX_RT_INTERVAL) {
      rt_torque_last = desired_torque;
      ts_torque_check_last = ts;
    }
  }

  // no torque if controls is not allowed
  if (!controls_allowed && (desired_torque != 0)) {
    violation = true;
  }

  // certain safety modes set their steer request bit low for one or more frame at a
  // predefined max frequency to avoid steering faults in certain situations
  bool steer_req_mismatch = (steer_req == 0) && (desired_torque != 0);
  if (!limits.has_steer_req_tolerance) {
    if (steer_req_mismatch) {
      violation = true;
    }

  } else {
    if (steer_req_mismatch) {
      if (invalid_steer_req_count == 0) {
        // disallow torque cut if not enough recent matching steer_req messages
        if (valid_steer_req_count < limits.min_valid_request_frames) {
          violation = true;
        }

        // or we've cut torque too recently in time
        uint32_t ts_elapsed = get_ts_elapsed(ts, ts_steer_req_mismatch_last);
        if (ts_elapsed < limits.min_valid_request_rt_interval) {
          violation = true;
        }
      } else {
        // or we're cutting more frames consecutively than allowed
        if (invalid_steer_req_count >= limits.max_invalid_request_frames) {
          violation = true;
        }
      }

      valid_steer_req_count = 0;
      ts_steer_req_mismatch_last = ts;
      invalid_steer_req_count = MIN(invalid_steer_req_count + 1, limits.max_invalid_request_frames);
    } else {
      valid_steer_req_count = MIN(valid_steer_req_count + 1, limits.min_valid_request_frames);
      invalid_steer_req_count = 0;
    }
  }

  // reset to 0 if either controls is not allowed or there's a violation
  if (violation || !controls_allowed) {
    valid_steer_req_count = 0;
    invalid_steer_req_count = 0;
    desired_torque_last = 0;
    rt_torque_last = 0;
    ts_torque_check_last = ts;
    ts_steer_req_mismatch_last = ts;
  }

  return violation;
}

static bool rt_angle_rate_limit_check(AngleSteeringLimits limits) {
  bool violation = false;
  uint32_t ts = microsecond_timer_get();

  // *** angle real time rate limit check ***
  int max_rt_msgs = ((float)limits.frequency * MAX_RT_INTERVAL / 1e6 * 1.2) + 1;  // 1.2x buffer
  if ((int)rt_angle_msgs > max_rt_msgs) {
    violation = true;
  }

  rt_angle_msgs += 1U;

  // every RT_INTERVAL reset message counter
  uint32_t ts_elapsed = get_ts_elapsed(ts, ts_angle_check_last);
  if (ts_elapsed >= MAX_RT_INTERVAL) {
    rt_angle_msgs = 0;
    ts_angle_check_last = ts;
  }

  return violation;
}

// Safety checks for angle-based steering commands
bool steer_angle_cmd_checks(int desired_angle, bool steer_control_enabled, const AngleSteeringLimits limits) {
  bool violation = false;

  if (controls_allowed && steer_control_enabled) {
    // convert floating point angle rate limits to integers in the scale of the desired angle on CAN,
    // add 1 to not false trigger the violation. also fudge the speed by 1 m/s so rate limits are
    // always slightly above openpilot's in case we read an updated speed in between angle commands
    // TODO: this speed fudge can be much lower, look at data to determine the lowest reasonable offset
    const float fudged_speed = (vehicle_speed.min / VEHICLE_SPEED_FACTOR) - 1.;
    int delta_angle_up = (interpolate(limits.angle_rate_up_lookup, fudged_speed) * limits.angle_deg_to_can) + 1.;
    int delta_angle_down = (interpolate(limits.angle_rate_down_lookup, fudged_speed) * limits.angle_deg_to_can) + 1.;

    // allow down limits at zero since small floats from openpilot will be rounded to 0
    // TODO: openpilot should be cognizant of this and not send small floats
    int highest_desired_angle = desired_angle_last + ((desired_angle_last > 0) ? delta_angle_up : delta_angle_down);
    int lowest_desired_angle = desired_angle_last - ((desired_angle_last >= 0) ? delta_angle_down : delta_angle_up);

    // check that commanded angle value isn't too far from measured, used to limit torque for some safety modes
    // ensure we start moving in direction of meas while respecting relaxed rate limits if error is exceeded
    if (limits.enforce_angle_error && ((vehicle_speed.values[0] / VEHICLE_SPEED_FACTOR) > limits.angle_error_min_speed)) {
      // flipped fudge to avoid false positives
      const float fudged_speed_error = (vehicle_speed.max / VEHICLE_SPEED_FACTOR) + 1.;
      const int delta_angle_up_relaxed = (interpolate(limits.angle_rate_up_lookup, fudged_speed_error) * limits.angle_deg_to_can) - 1.;
      const int delta_angle_down_relaxed = (interpolate(limits.angle_rate_down_lookup, fudged_speed_error) * limits.angle_deg_to_can) - 1.;

      // the minimum and maximum angle allowed based on the measured angle
      const int lowest_desired_angle_error = angle_meas.min - limits.max_angle_error - 1;
      const int highest_desired_angle_error = angle_meas.max + limits.max_angle_error + 1;

      // the MAX is to allow the desired angle to hit the edge of the bounds and not require going under it
      if (desired_angle_last > highest_desired_angle_error) {
        const int delta = (desired_angle_last >= 0) ? delta_angle_down_relaxed : delta_angle_up_relaxed;
        highest_desired_angle = MAX(desired_angle_last - delta, highest_desired_angle_error);

      } else if (desired_angle_last < lowest_desired_angle_error) {
        const int delta = (desired_angle_last <= 0) ? delta_angle_down_relaxed : delta_angle_up_relaxed;
        lowest_desired_angle = MIN(desired_angle_last + delta, lowest_desired_angle_error);

      } else {
        // already inside error boundary, don't allow commanding outside it
        highest_desired_angle = MIN(highest_desired_angle, highest_desired_angle_error);
        lowest_desired_angle = MAX(lowest_desired_angle, lowest_desired_angle_error);
      }

      // don't enforce above the max steer
      // TODO: this should always be done
      lowest_desired_angle = CLAMP(lowest_desired_angle, -limits.max_angle, limits.max_angle);
      highest_desired_angle = CLAMP(highest_desired_angle, -limits.max_angle, limits.max_angle);
    }

    // check not above ISO 11270 lateral accel assuming worst case road roll
    if (limits.angle_is_curvature) {

      // Limit to average banked road since safety doesn't have the roll
      static const float MAX_LATERAL_ACCEL = ISO_LATERAL_ACCEL - (EARTH_G * AVERAGE_ROAD_ROLL);  // ~2.4 m/s^2

      // Allow small tolerance by using minimum speed and rounding curvature up
      const float speed_lower = MAX(vehicle_speed.min / VEHICLE_SPEED_FACTOR, 1.0);
      const float speed_upper = MAX(vehicle_speed.max / VEHICLE_SPEED_FACTOR, 1.0);
      const int max_curvature_upper = (MAX_LATERAL_ACCEL / (speed_lower * speed_lower) * limits.angle_deg_to_can) + 1.;
      const int max_curvature_lower = (MAX_LATERAL_ACCEL / (speed_upper * speed_upper) * limits.angle_deg_to_can) - 1.;

      // ensure that the curvature error doesn't try to enforce above this limit
      if (desired_angle_last > 0) {
        lowest_desired_angle = CLAMP(lowest_desired_angle, -max_curvature_lower, max_curvature_lower);
        highest_desired_angle = CLAMP(highest_desired_angle, -max_curvature_upper, max_curvature_upper);
      } else {
        lowest_desired_angle = CLAMP(lowest_desired_angle, -max_curvature_upper, max_curvature_upper);
        highest_desired_angle = CLAMP(highest_desired_angle, -max_curvature_lower, max_curvature_lower);
      }
    }

    // check for violation;
    violation |= max_limit_check(desired_angle, highest_desired_angle, lowest_desired_angle);
  }
  desired_angle_last = desired_angle;

  // Angle should either be 0 or same as current angle while not steering
  if (!steer_control_enabled) {
    if (limits.inactive_angle_is_zero) {
      violation |= desired_angle != 0;
    } else {
      const int max_inactive_angle = CLAMP(angle_meas.max, -limits.max_angle, limits.max_angle) + 1;
      const int min_inactive_angle = CLAMP(angle_meas.min, -limits.max_angle, limits.max_angle) - 1;
      violation |= max_limit_check(desired_angle, max_inactive_angle, min_inactive_angle);
    }
  }

  // No angle control allowed when controls are not allowed
  if (!controls_allowed) {
    violation |= steer_control_enabled;
  }

  // reset to current angle if either controls is not allowed or there's a violation
  if (violation || !controls_allowed) {
    if (limits.inactive_angle_is_zero) {
      desired_angle_last = 0;
    } else {
      desired_angle_last = CLAMP(angle_meas.values[0], -limits.max_angle, limits.max_angle);
    }
  }

  return violation;
}

static float get_curvature_factor(const float speed, const AngleSteeringParams params) {
  // Matches VehicleModel.curvature_factor()
  return 1. / (1. - (params.slip_factor * (speed * speed))) / params.wheelbase;
}

static float get_angle_from_curvature(const float curvature, const float curvature_factor, const AngleSteeringParams params) {
  // Matches VehicleModel.get_steer_from_curvature()
  static const float RAD_TO_DEG = 57.29577951308232;
  return curvature * params.steer_ratio / curvature_factor * RAD_TO_DEG;
}

bool steer_angle_cmd_checks_vm(int desired_angle, bool steer_control_enabled, const AngleSteeringLimits limits,
                               const AngleSteeringParams params) {
  // This check uses a simple vehicle model to allow for constant lateral acceleration and jerk limits across all speeds.
  // TODO: remove the inaccurate breakpoint angle limiting function above and always use this one

  // Highway curves are rolled in the direction of the turn, add tolerance to compensate
  static const float MAX_LATERAL_ACCEL = ISO_LATERAL_ACCEL + (EARTH_G * AVERAGE_ROAD_ROLL);  // ~3.6 m/s^2
  // Lower than ISO 11270 lateral jerk limit, which is 5.0 m/s^3
  static const float MAX_LATERAL_JERK = 3.0 + (EARTH_G * AVERAGE_ROAD_ROLL);  // ~3.6 m/s^3

  const float fudged_speed = MAX((vehicle_speed.min / VEHICLE_SPEED_FACTOR) - 1.0, 1.0);
  const float curvature_factor = get_curvature_factor(fudged_speed, params);

  bool violation = false;

  if (controls_allowed && steer_control_enabled) {
    // *** ISO lateral jerk limit ***
    // calculate maximum angle rate per second
    const float max_curvature_rate_sec = MAX_LATERAL_JERK / (fudged_speed * fudged_speed);
    const float max_angle_rate_sec = get_angle_from_curvature(max_curvature_rate_sec, curvature_factor, params);

    // finally get max angle delta per frame
    const float max_angle_delta = max_angle_rate_sec / (float)limits.frequency;
    const int max_angle_delta_can = (max_angle_delta * limits.angle_deg_to_can) + 1.;

    // NOTE: symmetric up and down limits
    const int highest_desired_angle = desired_angle_last + max_angle_delta_can;
    const int lowest_desired_angle = desired_angle_last - max_angle_delta_can;

    violation |= max_limit_check(desired_angle, highest_desired_angle, lowest_desired_angle);

    // *** ISO lateral accel limit ***
    const float max_curvature = MAX_LATERAL_ACCEL / (fudged_speed * fudged_speed);
    const float max_angle = get_angle_from_curvature(max_curvature, curvature_factor, params);
    const int max_angle_can = (max_angle * limits.angle_deg_to_can) + 1.;

    violation |= max_limit_check(desired_angle, max_angle_can, -max_angle_can);

    // *** angle real time rate limit check ***
    violation |= rt_angle_rate_limit_check(limits);
  }
  desired_angle_last = desired_angle;

  // Angle should either be 0 or same as current angle while not steering
  if (!steer_control_enabled) {
    const int max_inactive_angle = CLAMP(angle_meas.max, -limits.max_angle, limits.max_angle) + 1;
    const int min_inactive_angle = CLAMP(angle_meas.min, -limits.max_angle, limits.max_angle) - 1;
    violation |= max_limit_check(desired_angle, max_inactive_angle, min_inactive_angle);
  }

  // No angle control allowed when controls are not allowed
  if (!controls_allowed) {
    violation |= steer_control_enabled;
  }

  // reset to current angle if either controls is not allowed or there's a violation
  if (violation || !controls_allowed) {
    desired_angle_last = CLAMP(angle_meas.values[0], -limits.max_angle, limits.max_angle);
  }

  return violation;
}
