// Define to prevent recursive inclusion
#ifndef CONFIG_H
#define CONFIG_H

#include "stm32f4xx_hal.h"

#define CORE_FREQ               96000000U // MCU frequency in hertz
#define PWM_FREQ                16000     // PWM frequency in Hz / is also used for buzzer
#define DEAD_TIME               48     // PWM deadtime
#define DELAY_IN_MAIN_LOOP      5     // in ms. default 5. it is independent of all the timing critical stuff. do not touch if you do not know what you are doing.
#define A2BIT_CONV              50     // A to bit for current conversion on ADC. Example: 1 A = 50, 2 A = 100, etc

#define ADC_CONV_CLOCK_CYCLES   (ADC_SAMPLETIME_15CYCLES)
#define ADC_CLOCK_DIV           (4)
#define ADC_TOTAL_CONV_TIME     (ADC_CLOCK_DIV * ADC_CONV_CLOCK_CYCLES) // = ((SystemCoreClock / ADC_CLOCK_HZ) * ADC_CONV_CLOCK_CYCLES), where ADC_CLOCK_HZ = SystemCoreClock/ADC_CLOCK_DIV

#define ANGLE_TO_DEGREES        0.021972656 // Convert 14 bit angle sensor output to degrees
#define GEARBOX_RATIO_LEFT      19
#define GEARBOX_RATIO_RIGHT     19
#define TRQ_LIMIT_LEFT          400      // Torque limit for knee gearbox(left)
#define TRQ_LIMIT_RIGHT         200      // Torque limit for hip gearbox(right)

#define KNEE_ADDR_OFFSET        0x100

#define BAT_FILT_COEF           655       // battery voltage filter coefficient in fixed-point. coef_fixedPoint = coef_floatingPoint * 2^16. In this case 655 = 0.01 * 2^16
#define BAT_CALIB_REAL_VOLTAGE  3192      // input voltage measured by multimeter (multiplied by 100). In this case 43.00 V * 100 = 4300
#define BAT_CALIB_ADC           1275      // adc-value measured by mainboard (value nr 5 on UART debug output)
#define BAT_CELLS               7         // battery number of cells. Normal Hoverboard battery: 10s
#define VOLTS_PER_PERCENT       0.00814    // Volts per percent, for conversion of volts to percentage
#define BAT_LVL2                (358 * BAT_CELLS * BAT_CALIB_ADC) / BAT_CALIB_REAL_VOLTAGE // 24%
#define BAT_LVL1                (351 * BAT_CELLS * BAT_CALIB_ADC) / BAT_CALIB_REAL_VOLTAGE // 15%
#define BAT_DEAD                (339 * BAT_CELLS * BAT_CALIB_ADC) / BAT_CALIB_REAL_VOLTAGE // 0%

#define TEMP_FILT_COEF          655       // temperature filter coefficient in fixed-point. coef_fixedPoint = coef_floatingPoint * 2^16. In this case 655 = 0.01 * 2^16
#define TEMP_CAL_LOW_ADC        945      // temperature 1: ADC value
#define TEMP_CAL_LOW_DEG_C      250       // temperature 1: measured temperature [°C * 10]. Here 35.8 °C
#define TEMP_CAL_HIGH_ADC       949      // temperature 2: ADC value
#define TEMP_CAL_HIGH_DEG_C     251       // temperature 2: measured temperature [°C * 10]. Here 48.9 °C
#define TEMP_WARNING_ENABLE     0         // to beep or not to beep, 1 or 0, DO NOT ACTIVITE WITHOUT CALIBRATION!
#define TEMP_WARNING            600       // annoying fast beeps [°C * 10].  Here 60.0 °C
#define TEMP_POWEROFF_ENABLE    0         // to poweroff or not to poweroff, 1 or 0, DO NOT ACTIVITE WITHOUT CALIBRATION!
#define TEMP_POWEROFF           650       // overheat poweroff. (while not driving) [°C * 10]. Here 65.0 °C

#define COM_CTRL        0               // [-] Commutation Control Type
#define SIN_CTRL        1               // [-] Sinusoidal Control Type
#define FOC_CTRL        2               // [-] Field Oriented Control (FOC) Type

#define OPEN_MODE       0               // [-] OPEN mode
#define VLT_MODE        1               // [-] VOLTAGE mode
#define SPD_MODE        2               // [-] SPEED mode
#define TRQ_MODE        3               // [-] TORQUE mode

// Enable/Disable Motor
#define MOTOR_LEFT_ENA                  // [-] Enable LEFT motor.  Comment-out if this motor is not needed to be operational
#define MOTOR_RIGHT_ENA                 // [-] Enable RIGHT motor. Comment-out if this motor is not needed to be operational

// Control selections
#define CTRL_TYP_SEL    FOC_CTRL        // [-] Control type selection: COM_CTRL, SIN_CTRL, FOC_CTRL (default)
#define CTRL_MOD_REQ    TRQ_MODE        // [-] Control mode request: OPEN_MODE, VLT_MODE (default), SPD_MODE, TRQ_MODE. Note: SPD_MODE and TRQ_MODE are only available for CTRL_FOC!
#define DIAG_ENA        1               // [-] Motor Diagnostics enable flag: 0 = Disabled, 1 = Enabled (default)

// Limitation settings
#define I_MOT_MAX       15              // [A] Maximum single motor current limit
#define I_DC_MAX        17              // [A] Maximum stage2 DC Link current limit for Commutation and Sinusoidal types (This is the final current protection. Above this value, current chopping is applied. To avoid this make sure that I_DC_MAX = I_MOT_MAX + 2A)
#define N_MOT_MAX       100            // [rpm] Maximum motor speed limit // 100 ~= 52m/m
#define TORQUE_BASE_MAX 1000

// Field Weakening / Phase Advance
#define FIELD_WEAK_ENA  0               // [-] Field Weakening / Phase Advance enable flag: 0 = Disabled (default), 1 = Enabled
#define FIELD_WEAK_MAX  5               // [A] Maximum Field Weakening D axis current (only for FOC). Higher current results in higher maximum speed. Up to 10A has been tested using 10" wheels.
#define PHASE_ADV_MAX   25              // [deg] Maximum Phase Advance angle (only for SIN). Higher angle results in higher maximum speed.
#define FIELD_WEAK_HI   1000            // (1000, 1500] Input target High threshold for reaching maximum Field Weakening / Phase Advance. Do NOT set this higher than 1500.
#define FIELD_WEAK_LO   750             // ( 500, 1000] Input target Low threshold for starting Field Weakening / Phase Advance. Do NOT set this higher than 1000.

#define SPEED_COEFFICIENT   16384 // Default for SPEED_COEFFICIENT 1.0f [-] higher value == stronger. [0, 65535] = [-2.0 - 2.0]. In this case 16384 = 1.0 * 2^14

#endif
