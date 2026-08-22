#ifndef BLDC_DEFS_H
#define BLDC_DEFS_H

#include "stm32h7xx.h" // For GPIO_TypeDef

#define COM_CTRL        0               // [-] Commutation Control Type
#define SIN_CTRL        1               // [-] Sinusoidal Control Type
#define FOC_CTRL        2               // [-] Field Oriented Control (FOC) Type

#define LEFT_TIM TIM8
#define RIGHT_TIM TIM1

#define PWM_FREQ 32000
#define PWM_MARGIN 100
#define CF_SPEED_COEF (PWM_FREQ / 3)
#define MAX_RPM 1000
#define RPM_TO_UNIT 16
#define RPM_DEADBAND 1

#define BODY_MOTOR_LEFT 1U
#define BODY_MOTOR_RIGHT 2U

// Stall detection recovery time. Time to wait after a stall before re-enabling controls.
#define STALL_DEQUAL_TIME_MS 3000
#define T_ERR_DEQUAL_CYCLES (uint16_t)(STALL_DEQUAL_TIME_MS * (PWM_FREQ / 2) / 1000)

// Motor
#define I_DC_MAX 6
#define I_MOT_MAX 6
#define A2BIT_CONV 310
#define N_MOT_MAX 1000

// Control selections
#define CTRL_TYP_SEL    FOC_CTRL        // [-] Control type selection: COM_CTRL, SIN_CTRL, FOC_CTRL (default)
#define CTRL_MOD_REQ    SPD_MODE        // [-] Control mode request: OPEN_MODE, VLT_MODE (default), SPD_MODE, TRQ_MODE. Note: SPD_MODE and TRQ_MODE are only available for CTRL_FOC!
#define DIAG_ENA        1               // [-] Motor Diagnostics enable flag: 0 = Disabled, 1 = Enabled (default)

// Field Weakening / Phase Advance
#define FIELD_WEAK_ENA  1               // [-] Field Weakening / Phase Advance enable flag: 0 = Disabled (default), 1 = Enabled
#define FIELD_WEAK_MAX  5               // [A] Maximum Field Weakening D axis current (only for FOC).
#define PHASE_ADV_MAX   25              // [deg] Maximum Phase Advance angle (only for SIN).
#define FIELD_WEAK_HI   1000            // Input target High threshold
#define FIELD_WEAK_LO   750             // Input target Low threshold

// Battery configuration
#define BAT_CELLS               3       // 3 sets in series
#define BAT_CELL_FULL_MV        4200U   // mV per cell at 100%
#define BAT_CELL_EMPTY_MV       3386U   // mV per cell at 0% (from V1: 4200 - 100 * 8.14)
#define VOLTS_PER_PERCENT       0.00814 // Volts per percent, for conversion of volts to percentage
#define BAT_CALIB_REAL_VOLTAGE  1260U   // multimeter voltage
#define BAT_CALIB_ADC           1275U   // adc reading voltage

void bldc_init(void);
void bldc_step(void);

#endif