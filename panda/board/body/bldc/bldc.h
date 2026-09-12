#ifndef BLDC_H
#define BLDC_H

#include "board/body/bldc/bldc_defs.h"
#include "board/body/boards/board_declarations.h"

#include <stdint.h>
#include <stdbool.h>

#include "board/stm32h7/lladc.h"

// Matlab includes and defines - from auto-code generation
#include "BLDC_controller.h"           /* Model's header file */
#include "BLDC_controller.c"
#include "BLDC_controller_data.c"
#include "rtwtypes.h"

static RT_MODEL rtM_Left_Obj;
static RT_MODEL rtM_Right_Obj;

RT_MODEL *const rtM_Left = &rtM_Left_Obj;
RT_MODEL *const rtM_Right = &rtM_Right_Obj;

static DW   rtDW_Left;                  /* Observable states */
static ExtU rtU_Left;                   /* External inputs */
static ExtY rtY_Left;                   /* External outputs */
extern P    rtP_Left; // This is defined in BLDC_controller_data.c

static DW   rtDW_Right;                 /* Observable states */
static ExtU rtU_Right;                  /* External inputs */
static ExtY rtY_Right;                  /* External outputs */
static P    rtP_Right;                  /* Parameters */

//static int16_t curDC_max = (I_DC_MAX * A2BIT_CONV);
static int16_t curL_phaA = 0, curL_phaC = 0, curL_DC = 0;
static int16_t curR_phaA = 0, curR_phaC = 0, curR_DC = 0;

volatile uint16_t batt_voltage_raw = 0;
volatile uint16_t batt_percentage = 0;

volatile int rpm_left = 0;
volatile int rpm_right = 0;

volatile bool enable_motors = 0;        // initially motors are disabled for safety
static bool enableFin = 0;

static const uint16_t pwm_res = ( (uint32_t)CORE_FREQ * 1000000U / 2U ) / PWM_FREQ;

static uint16_t offsetcount = 0;
static uint32_t offsetrlA = 0;
static uint32_t offsetrlC = 0;
static uint32_t offsetrrA = 0;
static uint32_t offsetrrC = 0;
static uint32_t offsetdcl = 0;
static uint32_t offsetdcr = 0;

#define ADC_CHANNEL_BLDC(a, c) {.adc = (a), .channel = (c), .sample_time = SAMPLETIME_16_CYCLES, .oversampling = OVERSAMPLING_4}

const adc_signal_t adc_curL_phaA = ADC_CHANNEL_BLDC(ADC2, 10);
const adc_signal_t adc_curL_phaC = ADC_CHANNEL_BLDC(ADC2, 11);
const adc_signal_t adc_curL_DC = ADC_CHANNEL_BLDC(ADC2, 18);
const adc_signal_t adc_curR_phaA = ADC_CHANNEL_BLDC(ADC1, 7);
const adc_signal_t adc_curR_phaC = ADC_CHANNEL_BLDC(ADC1, 15);
const adc_signal_t adc_curR_DC = ADC_CHANNEL_BLDC(ADC1, 5);
const adc_signal_t adc_batVoltage = ADC_CHANNEL_BLDC(ADC1, 4);

void motor_set_enable(bool enable) {
  enable_motors = enable;
}

float motor_encoder_get_speed_rpm(uint8_t motor) {
  float speed_rpm = 0.0f;
  if (motor == BODY_MOTOR_LEFT) {
    speed_rpm = (float)rtY_Left.n_mot;
  } else if (motor == BODY_MOTOR_RIGHT) {
    speed_rpm = (float)rtY_Right.n_mot;
  }

  if (ABS(speed_rpm) < RPM_DEADBAND) {
    speed_rpm = 0.0f;
  }

  return speed_rpm;
}

void bldc_init(void) {
  adc_init(ADC1);
  adc_init(ADC2);

  // Initialize Hall Sensors for Left Motor (PB6, PB7, PB8)
  set_gpio_mode(GPIOB, 6, MODE_INPUT); set_gpio_pullup(GPIOB, 6, PULL_UP);
  set_gpio_mode(GPIOB, 7, MODE_INPUT); set_gpio_pullup(GPIOB, 7, PULL_UP);
  set_gpio_mode(GPIOB, 8, MODE_INPUT); set_gpio_pullup(GPIOB, 8, PULL_UP);

  // Initialize Hall Sensors for Right Motor (PA0, PA1, PA2)
  set_gpio_mode(GPIOA, 0, MODE_INPUT); set_gpio_pullup(GPIOA, 0, PULL_UP);
  set_gpio_mode(GPIOA, 1, MODE_INPUT); set_gpio_pullup(GPIOA, 1, PULL_UP);
  set_gpio_mode(GPIOA, 2, MODE_INPUT); set_gpio_pullup(GPIOA, 2, PULL_UP);

  // Setup the model pointers for Left motor
  rtM_Left->defaultParam = &rtP_Left;
  rtM_Left->inputs = &rtU_Left;
  rtM_Left->outputs = &rtY_Left;
  rtM_Left->dwork = &rtDW_Left;
  BLDC_controller_initialize(rtM_Left);

  /* Set BLDC controller parameters */
  rtP_Left.b_angleMeasEna       = 0;            // Motor angle input: 0 = estimated angle, 1 = measured angle
  rtP_Left.z_selPhaCurMeasABC   = 2;            // Left motor measured current phases {Green, Blue} = {iA, iB}
  rtP_Left.z_ctrlTypSel         = CTRL_TYP_SEL;
  rtP_Left.b_diagEna            = DIAG_ENA;
  rtP_Left.i_max                = (int16_t)(I_MOT_MAX * A2BIT_CONV) << 4;
  rtP_Left.n_max                = N_MOT_MAX << 4;
  rtP_Left.b_fieldWeakEna       = FIELD_WEAK_ENA;
  rtP_Left.id_fieldWeakMax      = (int16_t)(FIELD_WEAK_MAX * A2BIT_CONV) << 4;
  rtP_Left.a_phaAdvMax          = PHASE_ADV_MAX << 4;
  rtP_Left.r_fieldWeakHi        = FIELD_WEAK_HI << 4;
  rtP_Left.r_fieldWeakLo        = FIELD_WEAK_LO << 4;
  rtP_Left.z_maxCntRst          = 4000;
  rtP_Left.cf_speedCoef         = CF_SPEED_COEF;
  rtP_Left.t_errQual            = 1280U; // 80ms at 16kHz loop rate
  rtP_Left.t_errDequal          = T_ERR_DEQUAL_CYCLES;

  // Setup the model pointers for Right motor
  rtP_Right = rtP_Left; // copy parameters
  // Right motor measured current phases A & C (based on adc_curR_phaA/C definitions)
  rtP_Right.z_selPhaCurMeasABC  = 2;
  rtM_Right->defaultParam = &rtP_Right;
  rtM_Right->inputs = &rtU_Right;
  rtM_Right->outputs = &rtY_Right;
  rtM_Right->dwork = &rtDW_Right;
  BLDC_controller_initialize(rtM_Right);

  // Initialize GPIOs for Motor Control
  // Left Motor (TIM1): PE8(CH1N), PE9(CH1), PE10(CH2N), PE11(CH2), PE12(CH3N), PE13(CH3)
  set_gpio_alternate(GPIOE, 8, GPIO_AF1_TIM1);
  set_gpio_alternate(GPIOE, 9, GPIO_AF1_TIM1);
  set_gpio_alternate(GPIOE, 10, GPIO_AF1_TIM1);
  set_gpio_alternate(GPIOE, 11, GPIO_AF1_TIM1);
  set_gpio_alternate(GPIOE, 12, GPIO_AF1_TIM1);
  set_gpio_alternate(GPIOE, 13, GPIO_AF1_TIM1);

  // Right Motor (TIM8): PC6(CH1), PC7(CH2), PC8(CH3), PA5(CH1N), PB14(CH2N), PB15(CH3N)
  set_gpio_alternate(GPIOC, 6, GPIO_AF3_TIM8);
  set_gpio_alternate(GPIOC, 7, GPIO_AF3_TIM8);
  set_gpio_alternate(GPIOC, 8, GPIO_AF3_TIM8);
  set_gpio_alternate(GPIOA, 5, GPIO_AF3_TIM8);
  set_gpio_alternate(GPIOB, 14, GPIO_AF3_TIM8);
  set_gpio_alternate(GPIOB, 15, GPIO_AF3_TIM8);

  // --- LEFT MOTOR (TIM1) ---
  // TIM8 is an advanced control timer. We configure it for 3-phase center-aligned PWM.
  LEFT_TIM->PSC = 0;
  LEFT_TIM->ARR = pwm_res;                              // Set auto-reload register for PWM_FREQ
  LEFT_TIM->CR1 = TIM_CR1_CMS_0;                        // Center-aligned mode 1
  LEFT_TIM->RCR = 1;                                    // Update event once per 2 PWM periods (16kHz)

  // Configure channel 1, 2, 3 as PWM mode 1, with preload enabled
  LEFT_TIM->CCMR1 = TIM_CCMR1_OC1M_2 | TIM_CCMR1_OC1M_1 | TIM_CCMR1_OC1PE | \
                    TIM_CCMR1_OC2M_2 | TIM_CCMR1_OC2M_1 | TIM_CCMR1_OC2PE;
  LEFT_TIM->CCMR2 = TIM_CCMR2_OC3M_2 | TIM_CCMR2_OC3M_1 | TIM_CCMR2_OC3PE;

  // Enable complementary outputs for channels 1, 2, 3
  LEFT_TIM->CCER = TIM_CCER_CC1E | TIM_CCER_CC1NE | \
                   TIM_CCER_CC2E | TIM_CCER_CC2NE | \
                   TIM_CCER_CC3E | TIM_CCER_CC3NE;

  // --- RIGHT MOTOR (TIM8) ---
  RIGHT_TIM->PSC = 0;
  RIGHT_TIM->ARR = pwm_res;
  RIGHT_TIM->CR1 = TIM_CR1_CMS_0;
  RIGHT_TIM->RCR = 1;

  RIGHT_TIM->CCMR1 = TIM_CCMR1_OC1M_2 | TIM_CCMR1_OC1M_1 | TIM_CCMR1_OC1PE | \
                     TIM_CCMR1_OC2M_2 | TIM_CCMR1_OC2M_1 | TIM_CCMR1_OC2PE;
  RIGHT_TIM->CCMR2 = TIM_CCMR2_OC3M_2 | TIM_CCMR2_OC3M_1 | TIM_CCMR2_OC3PE;

  RIGHT_TIM->CCER = TIM_CCER_CC1E | TIM_CCER_CC1NE | \
                    TIM_CCER_CC2E | TIM_CCER_CC2NE | \
                    TIM_CCER_CC3E | TIM_CCER_CC3NE;

  // Set dead time (20 cycles -> ~166ns with 120MHz clock) and enable motor outputs
  LEFT_TIM->BDTR = 20U | TIM_BDTR_MOE;
  RIGHT_TIM->BDTR = 20U | TIM_BDTR_MOE;

  // Generate an update event to load the registers
  LEFT_TIM->EGR = TIM_EGR_UG;
  RIGHT_TIM->EGR = TIM_EGR_UG;

  // Enable TIM8 update interrupt for bldc_step
  LEFT_TIM->DIER |= TIM_DIER_UIE;

  // Start the timers
  LEFT_TIM->CR1 |= TIM_CR1_CEN;
  RIGHT_TIM->CR1 |= TIM_CR1_CEN;
}

void bldc_step(void) {
  // Calibrate ADC offsets for the first few cycles
  if(offsetcount < 2000) {  // calibrate ADC offsets
    offsetcount++;
    uint32_t rawL_A = adc_get_raw(&adc_curL_phaA);
    uint32_t rawL_C = adc_get_raw(&adc_curL_phaC);
    uint32_t rawR_A = adc_get_raw(&adc_curR_phaA);
    uint32_t rawR_C = adc_get_raw(&adc_curR_phaC);
    uint32_t rawL_DC = adc_get_raw(&adc_curL_DC);
    uint32_t rawR_DC = adc_get_raw(&adc_curR_DC);

    if (offsetcount == 1) {
      offsetrlA = rawL_A; offsetrlC = rawL_C;
      offsetrrA = rawR_A; offsetrrC = rawR_C;
      offsetdcl = rawL_DC; offsetdcr = rawR_DC;
    } else {
      offsetrlA = (rawL_A + offsetrlA) / 2;
      offsetrlC = (rawL_C + offsetrlC) / 2;
      offsetrrA = (rawR_A + offsetrrA) / 2;
      offsetrrC = (rawR_C + offsetrrC) / 2;
      offsetdcl = (rawL_DC + offsetdcl) / 2;
      offsetdcr = (rawR_DC + offsetdcr) / 2;
    }
    return;
  }

  // Get Left motor currents
  curL_phaA = (int16_t)(((int32_t)offsetrlA - (int32_t)adc_get_raw(&adc_curL_phaA)) >> 5);
  curL_phaC = (int16_t)(((int32_t)offsetrlC - (int32_t)adc_get_raw(&adc_curL_phaC)) >> 5);
  curL_DC   = (int16_t)(((int32_t)offsetdcl - (int32_t)adc_get_raw(&adc_curL_DC)) >> 4);

  // Get Right motor currents
  curR_phaA = (int16_t)(((int32_t)offsetrrA - (int32_t)adc_get_raw(&adc_curR_phaA)) >> 5);
  curR_phaC = (int16_t)(((int32_t)offsetrrC - (int32_t)adc_get_raw(&adc_curR_phaC)) >> 5);
  curR_DC   = (int16_t)(((int32_t)offsetdcr - (int32_t)adc_get_raw(&adc_curR_DC)) >> 4);

  // Safety: Don't enable if offsets are bogus (e.g. ADC failed)
  if (offsetrrA == 0 || offsetrrC == 0 || !enable_motors) {
    enableFin = 0;
  } else {
    enableFin = 1;
  }

  // Read Hall Sensors
  rtU_Left.b_hallA = !((GPIOB->IDR >> 6) & 1);
  rtU_Left.b_hallB = !((GPIOB->IDR >> 7) & 1);
  rtU_Left.b_hallC = !((GPIOB->IDR >> 8) & 1);

  rtU_Right.b_hallA = !((GPIOA->IDR >> 0) & 1);
  rtU_Right.b_hallB = !((GPIOA->IDR >> 1) & 1);
  rtU_Right.b_hallC = !((GPIOA->IDR >> 2) & 1);

  if (!enableFin) {
    LEFT_TIM->BDTR &= ~TIM_BDTR_MOE;
    RIGHT_TIM->BDTR &= ~TIM_BDTR_MOE;
  } else {
    LEFT_TIM->BDTR |= TIM_BDTR_MOE;
    RIGHT_TIM->BDTR |= TIM_BDTR_MOE;
  }

  // read battery voltage
  batt_voltage_raw = adc_get_raw(&adc_batVoltage);

  int16_t batVoltageCalib = batt_voltage_raw * BAT_CALIB_REAL_VOLTAGE / BAT_CALIB_ADC;
  batt_percentage = 100 - (((420 * BAT_CELLS) - batVoltageCalib) / BAT_CELLS / VOLTS_PER_PERCENT / 100);

  // ========================= LEFT MOTOR ===========================
  rtU_Left.b_motEna      = enableFin;
  rtU_Left.z_ctrlModReq  = CTRL_MOD_REQ; // Speed Mode
  int deadband_rpm_left = rpm_left;
  if (ABS(deadband_rpm_left) < RPM_DEADBAND) {
    deadband_rpm_left = 0;
  }
  rtU_Left.r_inpTgt      = (CLAMP((int)deadband_rpm_left, -MAX_RPM, MAX_RPM) * RPM_TO_UNIT);

  rtU_Left.i_phaAB       = curL_phaA;
  rtU_Left.i_phaBC       = curL_phaC;
  rtU_Left.i_DCLink      = curL_DC;

  BLDC_controller_step(rtM_Left);

  int ul = rtY_Left.DC_phaA;
  int vl = rtY_Left.DC_phaB;
  int wl = rtY_Left.DC_phaC;

  LEFT_TIM->CCR1 = (uint16_t)CLAMP((ul + pwm_res / 2), PWM_MARGIN, pwm_res - PWM_MARGIN);
  LEFT_TIM->CCR2 = (uint16_t)CLAMP((vl + pwm_res / 2), PWM_MARGIN, pwm_res - PWM_MARGIN);
  LEFT_TIM->CCR3 = (uint16_t)CLAMP((wl + pwm_res / 2), PWM_MARGIN, pwm_res - PWM_MARGIN);

  // ========================= RIGHT MOTOR ===========================
  rtU_Right.b_motEna      = enableFin;
  rtU_Right.z_ctrlModReq  = CTRL_MOD_REQ; // Speed Mode
  int deadband_rpm_right = rpm_right;
  if (ABS(deadband_rpm_right) < RPM_DEADBAND) {
    deadband_rpm_right = 0;
  }
  rtU_Right.r_inpTgt      = -(CLAMP((int)deadband_rpm_right, -MAX_RPM, MAX_RPM) * RPM_TO_UNIT);

  rtU_Right.i_phaAB       = curR_phaA;
  rtU_Right.i_phaBC       = curR_phaC;
  rtU_Right.i_DCLink      = curR_DC;

  BLDC_controller_step(rtM_Right);

  int ur = rtY_Right.DC_phaA;
  int vr = rtY_Right.DC_phaB;
  int wr = rtY_Right.DC_phaC;

  RIGHT_TIM->CCR1 = (uint16_t)CLAMP((ur + pwm_res / 2), PWM_MARGIN, pwm_res - PWM_MARGIN);
  RIGHT_TIM->CCR2 = (uint16_t)CLAMP((vr + pwm_res / 2), PWM_MARGIN, pwm_res - PWM_MARGIN);
  RIGHT_TIM->CCR3 = (uint16_t)CLAMP((wr + pwm_res / 2), PWM_MARGIN, pwm_res - PWM_MARGIN);
}

#endif