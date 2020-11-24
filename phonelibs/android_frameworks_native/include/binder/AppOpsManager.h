/*
 * Copyright (C) 2013 The Android Open Source Project
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

#ifndef ANDROID_APP_OPS_MANAGER_H
#define ANDROID_APP_OPS_MANAGER_H

#include <binder/IAppOpsService.h>

#include <utils/threads.h>

// ---------------------------------------------------------------------------
namespace android {

class AppOpsManager
{
public:
    enum {
        MODE_ALLOWED = IAppOpsService::MODE_ALLOWED,
        MODE_IGNORED = IAppOpsService::MODE_IGNORED,
        MODE_ERRORED = IAppOpsService::MODE_ERRORED
    };

    enum {
        OP_NONE = -1,
        OP_COARSE_LOCATION = 0,
        OP_FINE_LOCATION = 1,
        OP_GPS = 2,
        OP_VIBRATE = 3,
        OP_READ_CONTACTS = 4,
        OP_WRITE_CONTACTS = 5,
        OP_READ_CALL_LOG = 6,
        OP_WRITE_CALL_LOG = 7,
        OP_READ_CALENDAR = 8,
        OP_WRITE_CALENDAR = 9,
        OP_WIFI_SCAN = 10,
        OP_POST_NOTIFICATION = 11,
        OP_NEIGHBORING_CELLS = 12,
        OP_CALL_PHONE = 13,
        OP_READ_SMS = 14,
        OP_WRITE_SMS = 15,
        OP_RECEIVE_SMS = 16,
        OP_RECEIVE_EMERGECY_SMS = 17,
        OP_RECEIVE_MMS = 18,
        OP_RECEIVE_WAP_PUSH = 19,
        OP_SEND_SMS = 20,
        OP_READ_ICC_SMS = 21,
        OP_WRITE_ICC_SMS = 22,
        OP_WRITE_SETTINGS = 23,
        OP_SYSTEM_ALERT_WINDOW = 24,
        OP_ACCESS_NOTIFICATIONS = 25,
        OP_CAMERA = 26,
        OP_RECORD_AUDIO = 27,
        OP_PLAY_AUDIO = 28,
        OP_READ_CLIPBOARD = 29,
        OP_WRITE_CLIPBOARD = 30,
        OP_TAKE_MEDIA_BUTTONS = 31,
        OP_TAKE_AUDIO_FOCUS = 32,
        OP_AUDIO_MASTER_VOLUME = 33,
        OP_AUDIO_VOICE_VOLUME = 34,
        OP_AUDIO_RING_VOLUME = 35,
        OP_AUDIO_MEDIA_VOLUME = 36,
        OP_AUDIO_ALARM_VOLUME = 37,
        OP_AUDIO_NOTIFICATION_VOLUME = 38,
        OP_AUDIO_BLUETOOTH_VOLUME = 39,
        OP_WAKE_LOCK = 40,
        OP_MONITOR_LOCATION = 41,
        OP_MONITOR_HIGH_POWER_LOCATION = 42,
        OP_GET_USAGE_STATS = 43,
        OP_MUTE_MICROPHONE = 44,
        OP_TOAST_WINDOW = 45,
        OP_PROJECT_MEDIA = 46,
        OP_ACTIVATE_VPN = 47,
        OP_WRITE_WALLPAPER = 48,
        OP_ASSIST_STRUCTURE = 49,
        OP_ASSIST_SCREENSHOT = 50,
        OP_READ_PHONE_STATE = 51,
        OP_ADD_VOICEMAIL = 52,
        OP_USE_SIP = 53,
        OP_PROCESS_OUTGOING_CALLS = 54,
        OP_USE_FINGERPRINT = 55,
        OP_BODY_SENSORS = 56,
        OP_READ_CELL_BROADCASTS = 57,
        OP_MOCK_LOCATION = 58,
        OP_READ_EXTERNAL_STORAGE = 59,
        OP_WRITE_EXTERNAL_STORAGE = 60,
        OP_TURN_SCREEN_ON = 61,
        OP_GET_ACCOUNTS = 62,
        OP_WIFI_CHANGE = 63,
        OP_BLUETOOTH_CHANGE = 64,
        OP_BOOT_COMPLETED = 65,
        OP_NFC_CHANGE = 66,
        OP_DATA_CONNECT_CHANGE = 67,
        OP_SU = 68
    };

    AppOpsManager();

    int32_t checkOp(int32_t op, int32_t uid, const String16& callingPackage);
    int32_t noteOp(int32_t op, int32_t uid, const String16& callingPackage);
    int32_t startOp(int32_t op, int32_t uid, const String16& callingPackage);
    void finishOp(int32_t op, int32_t uid, const String16& callingPackage);
    void startWatchingMode(int32_t op, const String16& packageName,
            const sp<IAppOpsCallback>& callback);
    void stopWatchingMode(const sp<IAppOpsCallback>& callback);
    int32_t permissionToOpCode(const String16& permission);

private:
    Mutex mLock;
    sp<IAppOpsService> mService;

    sp<IAppOpsService> getService();
};


}; // namespace android
// ---------------------------------------------------------------------------
#endif // ANDROID_APP_OPS_MANAGER_H
