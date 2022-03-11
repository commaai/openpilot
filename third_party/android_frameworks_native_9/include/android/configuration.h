/*
 * Copyright (C) 2010 The Android Open Source Project
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

/**
 * @addtogroup Configuration
 * @{
 */

/**
 * @file configuration.h
 */

#ifndef ANDROID_CONFIGURATION_H
#define ANDROID_CONFIGURATION_H

#include <sys/cdefs.h>

#include <android/asset_manager.h>

#ifdef __cplusplus
extern "C" {
#endif

struct AConfiguration;
/**
 * {@link AConfiguration} is an opaque type used to get and set
 * various subsystem configurations.
 *
 * A {@link AConfiguration} pointer can be obtained using:
 * - AConfiguration_new()
 * - AConfiguration_fromAssetManager()
 */
typedef struct AConfiguration AConfiguration;


/**
 * Define flags and constants for various subsystem configurations.
 */
enum {
    /** Orientation: not specified. */
    ACONFIGURATION_ORIENTATION_ANY  = 0x0000,
    /**
     * Orientation: value corresponding to the
     * <a href="@dacRoot/guide/topics/resources/providing-resources.html#OrientationQualifier">port</a>
     * resource qualifier.
     */
    ACONFIGURATION_ORIENTATION_PORT = 0x0001,
    /**
     * Orientation: value corresponding to the
     * <a href="@dacRoot/guide/topics/resources/providing-resources.html#OrientationQualifier">land</a>
     * resource qualifier.
     */
    ACONFIGURATION_ORIENTATION_LAND = 0x0002,
    /** @deprecated Not currently supported or used. */
    ACONFIGURATION_ORIENTATION_SQUARE = 0x0003,

    /** Touchscreen: not specified. */
    ACONFIGURATION_TOUCHSCREEN_ANY  = 0x0000,
    /**
     * Touchscreen: value corresponding to the
     * <a href="@dacRoot/guide/topics/resources/providing-resources.html#TouchscreenQualifier">notouch</a>
     * resource qualifier.
     */
    ACONFIGURATION_TOUCHSCREEN_NOTOUCH  = 0x0001,
    /** @deprecated Not currently supported or used. */
    ACONFIGURATION_TOUCHSCREEN_STYLUS  = 0x0002,
    /**
     * Touchscreen: value corresponding to the
     * <a href="@dacRoot/guide/topics/resources/providing-resources.html#TouchscreenQualifier">finger</a>
     * resource qualifier.
     */
    ACONFIGURATION_TOUCHSCREEN_FINGER  = 0x0003,

    /** Density: default density. */
    ACONFIGURATION_DENSITY_DEFAULT = 0,
    /**
     * Density: value corresponding to the
     * <a href="@dacRoot/guide/topics/resources/providing-resources.html#DensityQualifier">ldpi</a>
     * resource qualifier.
     */
    ACONFIGURATION_DENSITY_LOW = 120,
    /**
     * Density: value corresponding to the
     * <a href="@dacRoot/guide/topics/resources/providing-resources.html#DensityQualifier">mdpi</a>
     * resource qualifier.
     */
    ACONFIGURATION_DENSITY_MEDIUM = 160,
    /**
     * Density: value corresponding to the
     * <a href="@dacRoot/guide/topics/resources/providing-resources.html#DensityQualifier">tvdpi</a>
     * resource qualifier.
     */
    ACONFIGURATION_DENSITY_TV = 213,
    /**
     * Density: value corresponding to the
     * <a href="@dacRoot/guide/topics/resources/providing-resources.html#DensityQualifier">hdpi</a>
     * resource qualifier.
     */
    ACONFIGURATION_DENSITY_HIGH = 240,
    /**
     * Density: value corresponding to the
     * <a href="@dacRoot/guide/topics/resources/providing-resources.html#DensityQualifier">xhdpi</a>
     * resource qualifier.
     */
    ACONFIGURATION_DENSITY_XHIGH = 320,
    /**
     * Density: value corresponding to the
     * <a href="@dacRoot/guide/topics/resources/providing-resources.html#DensityQualifier">xxhdpi</a>
     * resource qualifier.
     */
    ACONFIGURATION_DENSITY_XXHIGH = 480,
    /**
     * Density: value corresponding to the
     * <a href="@dacRoot/guide/topics/resources/providing-resources.html#DensityQualifier">xxxhdpi</a>
     * resource qualifier.
     */
    ACONFIGURATION_DENSITY_XXXHIGH = 640,
    /** Density: any density. */
    ACONFIGURATION_DENSITY_ANY = 0xfffe,
    /** Density: no density specified. */
    ACONFIGURATION_DENSITY_NONE = 0xffff,

    /** Keyboard: not specified. */
    ACONFIGURATION_KEYBOARD_ANY  = 0x0000,
    /**
     * Keyboard: value corresponding to the
     * <a href="@dacRoot/guide/topics/resources/providing-resources.html#ImeQualifier">nokeys</a>
     * resource qualifier.
     */
    ACONFIGURATION_KEYBOARD_NOKEYS  = 0x0001,
    /**
     * Keyboard: value corresponding to the
     * <a href="@dacRoot/guide/topics/resources/providing-resources.html#ImeQualifier">qwerty</a>
     * resource qualifier.
     */
    ACONFIGURATION_KEYBOARD_QWERTY  = 0x0002,
    /**
     * Keyboard: value corresponding to the
     * <a href="@dacRoot/guide/topics/resources/providing-resources.html#ImeQualifier">12key</a>
     * resource qualifier.
     */
    ACONFIGURATION_KEYBOARD_12KEY  = 0x0003,

    /** Navigation: not specified. */
    ACONFIGURATION_NAVIGATION_ANY  = 0x0000,
    /**
     * Navigation: value corresponding to the
     * <a href="@@dacRoot/guide/topics/resources/providing-resources.html#NavigationQualifier">nonav</a>
     * resource qualifier.
     */
    ACONFIGURATION_NAVIGATION_NONAV  = 0x0001,
    /**
     * Navigation: value corresponding to the
     * <a href="@dacRoot/guide/topics/resources/providing-resources.html#NavigationQualifier">dpad</a>
     * resource qualifier.
     */
    ACONFIGURATION_NAVIGATION_DPAD  = 0x0002,
    /**
     * Navigation: value corresponding to the
     * <a href="@dacRoot/guide/topics/resources/providing-resources.html#NavigationQualifier">trackball</a>
     * resource qualifier.
     */
    ACONFIGURATION_NAVIGATION_TRACKBALL  = 0x0003,
    /**
     * Navigation: value corresponding to the
     * <a href="@dacRoot/guide/topics/resources/providing-resources.html#NavigationQualifier">wheel</a>
     * resource qualifier.
     */
    ACONFIGURATION_NAVIGATION_WHEEL  = 0x0004,

    /** Keyboard availability: not specified. */
    ACONFIGURATION_KEYSHIDDEN_ANY = 0x0000,
    /**
     * Keyboard availability: value corresponding to the
     * <a href="@dacRoot/guide/topics/resources/providing-resources.html#KeyboardAvailQualifier">keysexposed</a>
     * resource qualifier.
     */
    ACONFIGURATION_KEYSHIDDEN_NO = 0x0001,
    /**
     * Keyboard availability: value corresponding to the
     * <a href="@dacRoot/guide/topics/resources/providing-resources.html#KeyboardAvailQualifier">keyshidden</a>
     * resource qualifier.
     */
    ACONFIGURATION_KEYSHIDDEN_YES = 0x0002,
    /**
     * Keyboard availability: value corresponding to the
     * <a href="@dacRoot/guide/topics/resources/providing-resources.html#KeyboardAvailQualifier">keyssoft</a>
     * resource qualifier.
     */
    ACONFIGURATION_KEYSHIDDEN_SOFT = 0x0003,

    /** Navigation availability: not specified. */
    ACONFIGURATION_NAVHIDDEN_ANY = 0x0000,
    /**
     * Navigation availability: value corresponding to the
     * <a href="@dacRoot/guide/topics/resources/providing-resources.html#NavAvailQualifier">navexposed</a>
     * resource qualifier.
     */
    ACONFIGURATION_NAVHIDDEN_NO = 0x0001,
    /**
     * Navigation availability: value corresponding to the
     * <a href="@dacRoot/guide/topics/resources/providing-resources.html#NavAvailQualifier">navhidden</a>
     * resource qualifier.
     */
    ACONFIGURATION_NAVHIDDEN_YES = 0x0002,

    /** Screen size: not specified. */
    ACONFIGURATION_SCREENSIZE_ANY  = 0x00,
    /**
     * Screen size: value indicating the screen is at least
     * approximately 320x426 dp units, corresponding to the
     * <a href="@dacRoot/guide/topics/resources/providing-resources.html#ScreenSizeQualifier">small</a>
     * resource qualifier.
     */
    ACONFIGURATION_SCREENSIZE_SMALL = 0x01,
    /**
     * Screen size: value indicating the screen is at least
     * approximately 320x470 dp units, corresponding to the
     * <a href="@dacRoot/guide/topics/resources/providing-resources.html#ScreenSizeQualifier">normal</a>
     * resource qualifier.
     */
    ACONFIGURATION_SCREENSIZE_NORMAL = 0x02,
    /**
     * Screen size: value indicating the screen is at least
     * approximately 480x640 dp units, corresponding to the
     * <a href="@dacRoot/guide/topics/resources/providing-resources.html#ScreenSizeQualifier">large</a>
     * resource qualifier.
     */
    ACONFIGURATION_SCREENSIZE_LARGE = 0x03,
    /**
     * Screen size: value indicating the screen is at least
     * approximately 720x960 dp units, corresponding to the
     * <a href="@dacRoot/guide/topics/resources/providing-resources.html#ScreenSizeQualifier">xlarge</a>
     * resource qualifier.
     */
    ACONFIGURATION_SCREENSIZE_XLARGE = 0x04,

    /** Screen layout: not specified. */
    ACONFIGURATION_SCREENLONG_ANY = 0x00,
    /**
     * Screen layout: value that corresponds to the
     * <a href="@dacRoot/guide/topics/resources/providing-resources.html#ScreenAspectQualifier">notlong</a>
     * resource qualifier.
     */
    ACONFIGURATION_SCREENLONG_NO = 0x1,
    /**
     * Screen layout: value that corresponds to the
     * <a href="@dacRoot/guide/topics/resources/providing-resources.html#ScreenAspectQualifier">long</a>
     * resource qualifier.
     */
    ACONFIGURATION_SCREENLONG_YES = 0x2,

    ACONFIGURATION_SCREENROUND_ANY = 0x00,
    ACONFIGURATION_SCREENROUND_NO = 0x1,
    ACONFIGURATION_SCREENROUND_YES = 0x2,

    /** Wide color gamut: not specified. */
    ACONFIGURATION_WIDE_COLOR_GAMUT_ANY = 0x00,
    /**
     * Wide color gamut: value that corresponds to
     * <a href="@dacRoot/guide/topics/resources/providing-resources.html#WideColorGamutQualifier">no
     * nowidecg</a> resource qualifier specified.
     */
    ACONFIGURATION_WIDE_COLOR_GAMUT_NO = 0x1,
    /**
     * Wide color gamut: value that corresponds to
     * <a href="@dacRoot/guide/topics/resources/providing-resources.html#WideColorGamutQualifier">
     * widecg</a> resource qualifier specified.
     */
    ACONFIGURATION_WIDE_COLOR_GAMUT_YES = 0x2,

    /** HDR: not specified. */
    ACONFIGURATION_HDR_ANY = 0x00,
    /**
     * HDR: value that corresponds to
     * <a href="@dacRoot/guide/topics/resources/providing-resources.html#HDRQualifier">
     * lowdr</a> resource qualifier specified.
     */
    ACONFIGURATION_HDR_NO = 0x1,
    /**
     * HDR: value that corresponds to
     * <a href="@dacRoot/guide/topics/resources/providing-resources.html#HDRQualifier">
     * highdr</a> resource qualifier specified.
     */
    ACONFIGURATION_HDR_YES = 0x2,

    /** UI mode: not specified. */
    ACONFIGURATION_UI_MODE_TYPE_ANY = 0x00,
    /**
     * UI mode: value that corresponds to
     * <a href="@dacRoot/guide/topics/resources/providing-resources.html#UiModeQualifier">no
     * UI mode type</a> resource qualifier specified.
     */
    ACONFIGURATION_UI_MODE_TYPE_NORMAL = 0x01,
    /**
     * UI mode: value that corresponds to
     * <a href="@dacRoot/guide/topics/resources/providing-resources.html#UiModeQualifier">desk</a> resource qualifier specified.
     */
    ACONFIGURATION_UI_MODE_TYPE_DESK = 0x02,
    /**
     * UI mode: value that corresponds to
     * <a href="@dacRoot/guide/topics/resources/providing-resources.html#UiModeQualifier">car</a> resource qualifier specified.
     */
    ACONFIGURATION_UI_MODE_TYPE_CAR = 0x03,
    /**
     * UI mode: value that corresponds to
     * <a href="@dacRoot/guide/topics/resources/providing-resources.html#UiModeQualifier">television</a> resource qualifier specified.
     */
    ACONFIGURATION_UI_MODE_TYPE_TELEVISION = 0x04,
    /**
     * UI mode: value that corresponds to
     * <a href="@dacRoot/guide/topics/resources/providing-resources.html#UiModeQualifier">appliance</a> resource qualifier specified.
     */
    ACONFIGURATION_UI_MODE_TYPE_APPLIANCE = 0x05,
    /**
     * UI mode: value that corresponds to
     * <a href="@dacRoot/guide/topics/resources/providing-resources.html#UiModeQualifier">watch</a> resource qualifier specified.
     */
    ACONFIGURATION_UI_MODE_TYPE_WATCH = 0x06,
    /**
     * UI mode: value that corresponds to
     * <a href="@dacRoot/guide/topics/resources/providing-resources.html#UiModeQualifier">vr</a> resource qualifier specified.
     */
    ACONFIGURATION_UI_MODE_TYPE_VR_HEADSET = 0x07,

    /** UI night mode: not specified.*/
    ACONFIGURATION_UI_MODE_NIGHT_ANY = 0x00,
    /**
     * UI night mode: value that corresponds to
     * <a href="@dacRoot/guide/topics/resources/providing-resources.html#NightQualifier">notnight</a> resource qualifier specified.
     */
    ACONFIGURATION_UI_MODE_NIGHT_NO = 0x1,
    /**
     * UI night mode: value that corresponds to
     * <a href="@dacRoot/guide/topics/resources/providing-resources.html#NightQualifier">night</a> resource qualifier specified.
     */
    ACONFIGURATION_UI_MODE_NIGHT_YES = 0x2,

    /** Screen width DPI: not specified. */
    ACONFIGURATION_SCREEN_WIDTH_DP_ANY = 0x0000,

    /** Screen height DPI: not specified. */
    ACONFIGURATION_SCREEN_HEIGHT_DP_ANY = 0x0000,

    /** Smallest screen width DPI: not specified.*/
    ACONFIGURATION_SMALLEST_SCREEN_WIDTH_DP_ANY = 0x0000,

    /** Layout direction: not specified. */
    ACONFIGURATION_LAYOUTDIR_ANY  = 0x00,
    /**
     * Layout direction: value that corresponds to
     * <a href="@dacRoot/guide/topics/resources/providing-resources.html#LayoutDirectionQualifier">ldltr</a> resource qualifier specified.
     */
    ACONFIGURATION_LAYOUTDIR_LTR  = 0x01,
    /**
     * Layout direction: value that corresponds to
     * <a href="@dacRoot/guide/topics/resources/providing-resources.html#LayoutDirectionQualifier">ldrtl</a> resource qualifier specified.
     */
    ACONFIGURATION_LAYOUTDIR_RTL  = 0x02,

    /**
     * Bit mask for
     * <a href="@dacRoot/guide/topics/resources/providing-resources.html#MccQualifier">mcc</a>
     * configuration.
     */
    ACONFIGURATION_MCC = 0x0001,
    /**
     * Bit mask for
     * <a href="@dacRoot/guide/topics/resources/providing-resources.html#MccQualifier">mnc</a>
     * configuration.
     */
    ACONFIGURATION_MNC = 0x0002,
    /**
     * Bit mask for
     * <a href="{@docRoot}guide/topics/resources/providing-resources.html#LocaleQualifier">locale</a>
     * configuration.
     */
    ACONFIGURATION_LOCALE = 0x0004,
    /**
     * Bit mask for
     * <a href="@dacRoot/guide/topics/resources/providing-resources.html#TouchscreenQualifier">touchscreen</a>
     * configuration.
     */
    ACONFIGURATION_TOUCHSCREEN = 0x0008,
    /**
     * Bit mask for
     * <a href="@dacRoot/guide/topics/resources/providing-resources.html#ImeQualifier">keyboard</a>
     * configuration.
     */
    ACONFIGURATION_KEYBOARD = 0x0010,
    /**
     * Bit mask for
     * <a href="@dacRoot/guide/topics/resources/providing-resources.html#KeyboardAvailQualifier">keyboardHidden</a>
     * configuration.
     */
    ACONFIGURATION_KEYBOARD_HIDDEN = 0x0020,
    /**
     * Bit mask for
     * <a href="@dacRoot/guide/topics/resources/providing-resources.html#NavigationQualifier">navigation</a>
     * configuration.
     */
    ACONFIGURATION_NAVIGATION = 0x0040,
    /**
     * Bit mask for
     * <a href="@dacRoot/guide/topics/resources/providing-resources.html#OrientationQualifier">orientation</a>
     * configuration.
     */
    ACONFIGURATION_ORIENTATION = 0x0080,
    /**
     * Bit mask for
     * <a href="@dacRoot/guide/topics/resources/providing-resources.html#DensityQualifier">density</a>
     * configuration.
     */
    ACONFIGURATION_DENSITY = 0x0100,
    /**
     * Bit mask for
     * <a href="@dacRoot/guide/topics/resources/providing-resources.html#ScreenSizeQualifier">screen size</a>
     * configuration.
     */
    ACONFIGURATION_SCREEN_SIZE = 0x0200,
    /**
     * Bit mask for
     * <a href="@dacRoot/guide/topics/resources/providing-resources.html#VersionQualifier">platform version</a>
     * configuration.
     */
    ACONFIGURATION_VERSION = 0x0400,
    /**
     * Bit mask for screen layout configuration.
     */
    ACONFIGURATION_SCREEN_LAYOUT = 0x0800,
    /**
     * Bit mask for
     * <a href="@dacRoot/guide/topics/resources/providing-resources.html#UiModeQualifier">ui mode</a>
     * configuration.
     */
    ACONFIGURATION_UI_MODE = 0x1000,
    /**
     * Bit mask for
     * <a href="@dacRoot/guide/topics/resources/providing-resources.html#SmallestScreenWidthQualifier">smallest screen width</a>
     * configuration.
     */
    ACONFIGURATION_SMALLEST_SCREEN_SIZE = 0x2000,
    /**
     * Bit mask for
     * <a href="@dacRoot/guide/topics/resources/providing-resources.html#LayoutDirectionQualifier">layout direction</a>
     * configuration.
     */
    ACONFIGURATION_LAYOUTDIR = 0x4000,
    ACONFIGURATION_SCREEN_ROUND = 0x8000,
    /**
     * Bit mask for
     * <a href="@dacRoot/guide/topics/resources/providing-resources.html#WideColorGamutQualifier">wide color gamut</a>
     * and <a href="@dacRoot/guide/topics/resources/providing-resources.html#HDRQualifier">HDR</a> configurations.
     */
    ACONFIGURATION_COLOR_MODE = 0x10000,
    /**
     * Constant used to to represent MNC (Mobile Network Code) zero.
     * 0 cannot be used, since it is used to represent an undefined MNC.
     */
    ACONFIGURATION_MNC_ZERO = 0xffff,
};

/**
 * Create a new AConfiguration, initialized with no values set.
 */
AConfiguration* AConfiguration_new();

/**
 * Free an AConfiguration that was previously created with
 * AConfiguration_new().
 */
void AConfiguration_delete(AConfiguration* config);

/**
 * Create and return a new AConfiguration based on the current configuration in
 * use in the given {@link AAssetManager}.
 */
void AConfiguration_fromAssetManager(AConfiguration* out, AAssetManager* am);

/**
 * Copy the contents of 'src' to 'dest'.
 */
void AConfiguration_copy(AConfiguration* dest, AConfiguration* src);

/**
 * Return the current MCC set in the configuration.  0 if not set.
 */
int32_t AConfiguration_getMcc(AConfiguration* config);

/**
 * Set the current MCC in the configuration.  0 to clear.
 */
void AConfiguration_setMcc(AConfiguration* config, int32_t mcc);

/**
 * Return the current MNC set in the configuration.  0 if not set.
 */
int32_t AConfiguration_getMnc(AConfiguration* config);

/**
 * Set the current MNC in the configuration.  0 to clear.
 */
void AConfiguration_setMnc(AConfiguration* config, int32_t mnc);

/**
 * Return the current language code set in the configuration.  The output will
 * be filled with an array of two characters.  They are not 0-terminated.  If
 * a language is not set, they will be 0.
 */
void AConfiguration_getLanguage(AConfiguration* config, char* outLanguage);

/**
 * Set the current language code in the configuration, from the first two
 * characters in the string.
 */
void AConfiguration_setLanguage(AConfiguration* config, const char* language);

/**
 * Return the current country code set in the configuration.  The output will
 * be filled with an array of two characters.  They are not 0-terminated.  If
 * a country is not set, they will be 0.
 */
void AConfiguration_getCountry(AConfiguration* config, char* outCountry);

/**
 * Set the current country code in the configuration, from the first two
 * characters in the string.
 */
void AConfiguration_setCountry(AConfiguration* config, const char* country);

/**
 * Return the current ACONFIGURATION_ORIENTATION_* set in the configuration.
 */
int32_t AConfiguration_getOrientation(AConfiguration* config);

/**
 * Set the current orientation in the configuration.
 */
void AConfiguration_setOrientation(AConfiguration* config, int32_t orientation);

/**
 * Return the current ACONFIGURATION_TOUCHSCREEN_* set in the configuration.
 */
int32_t AConfiguration_getTouchscreen(AConfiguration* config);

/**
 * Set the current touchscreen in the configuration.
 */
void AConfiguration_setTouchscreen(AConfiguration* config, int32_t touchscreen);

/**
 * Return the current ACONFIGURATION_DENSITY_* set in the configuration.
 */
int32_t AConfiguration_getDensity(AConfiguration* config);

/**
 * Set the current density in the configuration.
 */
void AConfiguration_setDensity(AConfiguration* config, int32_t density);

/**
 * Return the current ACONFIGURATION_KEYBOARD_* set in the configuration.
 */
int32_t AConfiguration_getKeyboard(AConfiguration* config);

/**
 * Set the current keyboard in the configuration.
 */
void AConfiguration_setKeyboard(AConfiguration* config, int32_t keyboard);

/**
 * Return the current ACONFIGURATION_NAVIGATION_* set in the configuration.
 */
int32_t AConfiguration_getNavigation(AConfiguration* config);

/**
 * Set the current navigation in the configuration.
 */
void AConfiguration_setNavigation(AConfiguration* config, int32_t navigation);

/**
 * Return the current ACONFIGURATION_KEYSHIDDEN_* set in the configuration.
 */
int32_t AConfiguration_getKeysHidden(AConfiguration* config);

/**
 * Set the current keys hidden in the configuration.
 */
void AConfiguration_setKeysHidden(AConfiguration* config, int32_t keysHidden);

/**
 * Return the current ACONFIGURATION_NAVHIDDEN_* set in the configuration.
 */
int32_t AConfiguration_getNavHidden(AConfiguration* config);

/**
 * Set the current nav hidden in the configuration.
 */
void AConfiguration_setNavHidden(AConfiguration* config, int32_t navHidden);

/**
 * Return the current SDK (API) version set in the configuration.
 */
int32_t AConfiguration_getSdkVersion(AConfiguration* config);

/**
 * Set the current SDK version in the configuration.
 */
void AConfiguration_setSdkVersion(AConfiguration* config, int32_t sdkVersion);

/**
 * Return the current ACONFIGURATION_SCREENSIZE_* set in the configuration.
 */
int32_t AConfiguration_getScreenSize(AConfiguration* config);

/**
 * Set the current screen size in the configuration.
 */
void AConfiguration_setScreenSize(AConfiguration* config, int32_t screenSize);

/**
 * Return the current ACONFIGURATION_SCREENLONG_* set in the configuration.
 */
int32_t AConfiguration_getScreenLong(AConfiguration* config);

/**
 * Set the current screen long in the configuration.
 */
void AConfiguration_setScreenLong(AConfiguration* config, int32_t screenLong);

/**
 * Return the current ACONFIGURATION_SCREENROUND_* set in the configuration.
 */
int32_t AConfiguration_getScreenRound(AConfiguration* config);

/**
 * Set the current screen round in the configuration.
 */
void AConfiguration_setScreenRound(AConfiguration* config, int32_t screenRound);

/**
 * Return the current ACONFIGURATION_UI_MODE_TYPE_* set in the configuration.
 */
int32_t AConfiguration_getUiModeType(AConfiguration* config);

/**
 * Set the current UI mode type in the configuration.
 */
void AConfiguration_setUiModeType(AConfiguration* config, int32_t uiModeType);

/**
 * Return the current ACONFIGURATION_UI_MODE_NIGHT_* set in the configuration.
 */
int32_t AConfiguration_getUiModeNight(AConfiguration* config);

/**
 * Set the current UI mode night in the configuration.
 */
void AConfiguration_setUiModeNight(AConfiguration* config, int32_t uiModeNight);

#if __ANDROID_API__ >= 13
/**
 * Return the current configuration screen width in dp units, or
 * ACONFIGURATION_SCREEN_WIDTH_DP_ANY if not set.
 */
int32_t AConfiguration_getScreenWidthDp(AConfiguration* config);

/**
 * Set the configuration's current screen width in dp units.
 */
void AConfiguration_setScreenWidthDp(AConfiguration* config, int32_t value);

/**
 * Return the current configuration screen height in dp units, or
 * ACONFIGURATION_SCREEN_HEIGHT_DP_ANY if not set.
 */
int32_t AConfiguration_getScreenHeightDp(AConfiguration* config);

/**
 * Set the configuration's current screen width in dp units.
 */
void AConfiguration_setScreenHeightDp(AConfiguration* config, int32_t value);

/**
 * Return the configuration's smallest screen width in dp units, or
 * ACONFIGURATION_SMALLEST_SCREEN_WIDTH_DP_ANY if not set.
 */
int32_t AConfiguration_getSmallestScreenWidthDp(AConfiguration* config);

/**
 * Set the configuration's smallest screen width in dp units.
 */
void AConfiguration_setSmallestScreenWidthDp(AConfiguration* config, int32_t value);
#endif /* __ANDROID_API__ >= 13 */

#if __ANDROID_API__ >= 17
/**
 * Return the configuration's layout direction, or
 * ACONFIGURATION_LAYOUTDIR_ANY if not set.
 */
int32_t AConfiguration_getLayoutDirection(AConfiguration* config);

/**
 * Set the configuration's layout direction.
 */
void AConfiguration_setLayoutDirection(AConfiguration* config, int32_t value);
#endif /* __ANDROID_API__ >= 17 */

/**
 * Perform a diff between two configurations.  Returns a bit mask of
 * ACONFIGURATION_* constants, each bit set meaning that configuration element
 * is different between them.
 */
int32_t AConfiguration_diff(AConfiguration* config1, AConfiguration* config2);

/**
 * Determine whether 'base' is a valid configuration for use within the
 * environment 'requested'.  Returns 0 if there are any values in 'base'
 * that conflict with 'requested'.  Returns 1 if it does not conflict.
 */
int32_t AConfiguration_match(AConfiguration* base, AConfiguration* requested);

/**
 * Determine whether the configuration in 'test' is better than the existing
 * configuration in 'base'.  If 'requested' is non-NULL, this decision is based
 * on the overall configuration given there.  If it is NULL, this decision is
 * simply based on which configuration is more specific.  Returns non-0 if
 * 'test' is better than 'base'.
 *
 * This assumes you have already filtered the configurations with
 * AConfiguration_match().
 */
int32_t AConfiguration_isBetterThan(AConfiguration* base, AConfiguration* test,
        AConfiguration* requested);

#ifdef __cplusplus
};
#endif

#endif // ANDROID_CONFIGURATION_H

/** @} */
