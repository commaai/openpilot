/*
 * Copyright (C) 2013 The Android Open Source Project
 * Copyright (C) 2015 The CyanogenMod Project
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef ANDROID_BATTERYSERVICE_H
#define ANDROID_BATTERYSERVICE_H

#include <binder/Parcel.h>
#include <sys/types.h>
#include <utils/Errors.h>
#include <utils/String8.h>

namespace android {

// must be kept in sync with definitions in BatteryManager.java
enum {
    BATTERY_STATUS_UNKNOWN = 1, // equals BatteryManager.BATTERY_STATUS_UNKNOWN constant
    BATTERY_STATUS_CHARGING = 2, // equals BatteryManager.BATTERY_STATUS_CHARGING constant
    BATTERY_STATUS_DISCHARGING = 3, // equals BatteryManager.BATTERY_STATUS_DISCHARGING constant
    BATTERY_STATUS_NOT_CHARGING = 4, // equals BatteryManager.BATTERY_STATUS_NOT_CHARGING constant
    BATTERY_STATUS_FULL = 5, // equals BatteryManager.BATTERY_STATUS_FULL constant
};

// must be kept in sync with definitions in BatteryManager.java
enum {
    BATTERY_HEALTH_UNKNOWN = 1, // equals BatteryManager.BATTERY_HEALTH_UNKNOWN constant
    BATTERY_HEALTH_GOOD = 2, // equals BatteryManager.BATTERY_HEALTH_GOOD constant
    BATTERY_HEALTH_OVERHEAT = 3, // equals BatteryManager.BATTERY_HEALTH_OVERHEAT constant
    BATTERY_HEALTH_DEAD = 4, // equals BatteryManager.BATTERY_HEALTH_DEAD constant
    BATTERY_HEALTH_OVER_VOLTAGE = 5, // equals BatteryManager.BATTERY_HEALTH_OVER_VOLTAGE constant
    BATTERY_HEALTH_UNSPECIFIED_FAILURE = 6, // equals BatteryManager.BATTERY_HEALTH_UNSPECIFIED_FAILURE constant
    BATTERY_HEALTH_COLD = 7, // equals BatteryManager.BATTERY_HEALTH_COLD constant
};

// must be kept in sync with definitions in BatteryProperty.java
enum {
    BATTERY_PROP_CHARGE_COUNTER = 1, // equals BatteryProperty.CHARGE_COUNTER constant
    BATTERY_PROP_CURRENT_NOW = 2, // equals BatteryProperty.CURRENT_NOW constant
    BATTERY_PROP_CURRENT_AVG = 3, // equals BatteryProperty.CURRENT_AVG constant
    BATTERY_PROP_CAPACITY = 4, // equals BatteryProperty.CAPACITY constant
    BATTERY_PROP_ENERGY_COUNTER = 5, // equals BatteryProperty.ENERGY_COUNTER constant
};

struct BatteryProperties {
    bool chargerAcOnline;
    bool chargerUsbOnline;
    bool chargerWirelessOnline;
    int maxChargingCurrent;
    int batteryStatus;
    int batteryHealth;
    bool batteryPresent;
    int batteryLevel;
    int batteryVoltage;
    int batteryTemperature;
    String8 batteryTechnology;

    bool dockBatterySupported;
    bool chargerDockAcOnline;
    int dockBatteryStatus;
    int dockBatteryHealth;
    bool dockBatteryPresent;
    int dockBatteryLevel;
    int dockBatteryVoltage;
    int dockBatteryTemperature;
    String8 dockBatteryTechnology;

    status_t writeToParcel(Parcel* parcel) const;
    status_t readFromParcel(Parcel* parcel);
};

struct BatteryProperty {
    int64_t valueInt64;

    status_t writeToParcel(Parcel* parcel) const;
    status_t readFromParcel(Parcel* parcel);
};

}; // namespace android

#endif // ANDROID_BATTERYSERVICE_H
