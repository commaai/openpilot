#include <stdint.h>
#include "stm32f4xx_hal.h"
#include "defines.h"
#include "config.h"
#include "util.h"
#include "bldc/BLDC_controller.h"
#include "bldc/rtwtypes.h"

// TODO: simplify util.c to util.h

//------------------------------------------------------------------------
// Global variables set externally
//------------------------------------------------------------------------
extern int16_t batVoltage;
extern uint8_t buzzerCount;             // global variable for the buzzer counts. can be 1, 2, 3, 4, 5, 6, 7...
extern uint8_t buzzerFreq;              // global variable for the buzzer pitch. can be 1, 2, 3, 4, 5, 6, 7...
extern uint8_t buzzerPattern;           // global variable for the buzzer pattern. can be 1, 2, 3, 4, 5, 6, 7...

extern uint8_t enable_motors;           // global variable for motor enable
extern uint8_t ignition;                // global variable for ignition on SBU2 line

extern uint8_t hw_type;

extern board_t board;

//------------------------------------------------------------------------
// Matlab defines - from auto-code generation
//---------------
RT_MODEL rtM_Left_;                     /* Real-time model */
RT_MODEL rtM_Right_;                    /* Real-time model */
RT_MODEL *const rtM_Left  = &rtM_Left_;
RT_MODEL *const rtM_Right = &rtM_Right_;

extern P rtP_Left;                      /* Block parameters (auto storage) */
DW       rtDW_Left;                     /* Observable states */
ExtU     rtU_Left;                      /* External inputs */
ExtY     rtY_Left;                      /* External outputs */

P        rtP_Right;                     /* Block parameters (auto storage) */
DW       rtDW_Right;                    /* Observable states */
ExtU     rtU_Right;                     /* External inputs */
ExtY     rtY_Right;                     /* External outputs */
//---------------

int16_t  speedAvg;                      // average measured speed
int16_t  speedAvgAbs;                   // average measured speed in absolute

uint8_t  ctrlModReqRaw = CTRL_MOD_REQ;
uint8_t  ctrlModReq    = CTRL_MOD_REQ;  // Final control mode request


void BLDC_Init(void) {
  /* Set BLDC controller parameters */
  rtP_Left.b_angleMeasEna       = 0;            // Motor angle input: 0 = estimated angle, 1 = measured angle (e.g. if encoder is available)
  rtP_Left.z_selPhaCurMeasABC   = 0;            // Left motor measured current phases {Green, Blue} = {iA, iB} -> do NOT change
  rtP_Left.z_ctrlTypSel         = CTRL_TYP_SEL;
  rtP_Left.b_diagEna            = DIAG_ENA;
  rtP_Left.i_max                = (I_MOT_MAX * A2BIT_CONV) << 4;        // fixdt(1,16,4)
  rtP_Left.n_max                = N_MOT_MAX << 4;                       // fixdt(1,16,4)
  rtP_Left.b_fieldWeakEna       = FIELD_WEAK_ENA;
  rtP_Left.id_fieldWeakMax      = (FIELD_WEAK_MAX * A2BIT_CONV) << 4;   // fixdt(1,16,4)
  rtP_Left.a_phaAdvMax          = PHASE_ADV_MAX << 4;                   // fixdt(1,16,4)
  rtP_Left.r_fieldWeakHi        = FIELD_WEAK_HI << 4;                   // fixdt(1,16,4)
  rtP_Left.r_fieldWeakLo        = FIELD_WEAK_LO << 4;                   // fixdt(1,16,4)

  rtP_Right                     = rtP_Left;     // Copy the Left motor parameters to the Right motor parameters
  rtP_Right.n_max               = N_MOT_MAX << 4; // But add separate max RPM limit
  rtP_Right.z_selPhaCurMeasABC  = 1;            // Right motor measured current phases {Blue, Yellow} = {iB, iC} -> do NOT change

  /* Pack LEFT motor data into RTM */
  rtM_Left->defaultParam        = &rtP_Left;
  rtM_Left->dwork               = &rtDW_Left;
  rtM_Left->inputs              = &rtU_Left;
  rtM_Left->outputs             = &rtY_Left;

  /* Pack RIGHT motor data into RTM */
  rtM_Right->defaultParam       = &rtP_Right;
  rtM_Right->dwork              = &rtDW_Right;
  rtM_Right->inputs             = &rtU_Right;
  rtM_Right->outputs            = &rtY_Right;

  /* Initialize BLDC controllers */
  BLDC_controller_initialize(rtM_Left);
  BLDC_controller_initialize(rtM_Right);
}

void out_enable(uint8_t out, bool enabled) {
  switch(out) {
    case LED_GREEN:
      HAL_GPIO_WritePin(board.led_portG, board.led_pinG, !enabled);
      break;
    case LED_RED:
      HAL_GPIO_WritePin(board.led_portR, board.led_pinR, !enabled);
      break;
    case LED_BLUE:
      HAL_GPIO_WritePin(board.led_portB, board.led_pinB, !enabled);
      break;
    case IGNITION:
      HAL_GPIO_WritePin(board.ignition_port, board.ignition_pin, enabled);
      break;
    case POWERSWITCH:
      HAL_GPIO_WritePin(OFF_PORT, OFF_PIN, enabled);
      break;
    case TRANSCEIVER:
      HAL_GPIO_WritePin(board.can_portEN, board.can_pinEN, !enabled);
      break;
  }
}

void poweronMelody(void) {
    buzzerCount = 0;
    for (int i = 8; i >= 0; i--) {
      buzzerFreq = (uint8_t)i;
      HAL_Delay(100);
    }
    buzzerFreq = 0;
}

void beepCount(uint8_t cnt, uint8_t freq, uint8_t pattern) {
    buzzerCount   = cnt;
    buzzerFreq    = freq;
    buzzerPattern = pattern;
}

void beepLong(uint8_t freq) {
    buzzerCount = 0;
    buzzerFreq = freq;
    HAL_Delay(500);
    buzzerFreq = 0;
}

void beepShort(uint8_t freq) {
    buzzerCount = 0;
    buzzerFreq = freq;
    HAL_Delay(100);
    buzzerFreq = 0;
}

void beepShortMany(uint8_t cnt, int8_t dir) {
    if (dir >= 0) {   // increasing tone
      for(uint8_t i = 2*cnt; i >= 2; i=i-2) {
        beepShort(i + 3);
      }
    } else {          // decreasing tone
      for(uint8_t i = 2; i <= 2*cnt; i=i+2) {
        beepShort(i + 3);
      }
    }
}

void calcAvgSpeed(void) {
    // Calculate measured average speed. The minus sign (-) is because motors spin in opposite directions
      speedAvg    = ( rtY_Left.n_mot - rtY_Right.n_mot) / 2;

    // Handle the case when SPEED_COEFFICIENT sign is negative (which is when most significant bit is 1)
    if (SPEED_COEFFICIENT & (1 << 16)) {
      speedAvg    = -speedAvg;
    }
    speedAvgAbs   = ABS(speedAvg);
}

void poweroff(void) {
  enable_motors = 0;
  buzzerCount = 0;
  buzzerPattern = 0;
  for (int i = 0; i < 8; i++) {
    buzzerFreq = (uint8_t)i;
    HAL_Delay(100);
  }
  out_enable(POWERSWITCH, false);
  while(1) {
    // Temporarily, to see that we went to power off but can't switch the latch
    HAL_GPIO_TogglePin(board.led_portR, board.led_pinR);
    HAL_Delay(100);
  }
}

void poweroffPressCheck(void) {
  if(HAL_GPIO_ReadPin(BUTTON_PORT, BUTTON_PIN)) {
    uint16_t cnt_press = 0;
    while(HAL_GPIO_ReadPin(BUTTON_PORT, BUTTON_PIN)) {
      HAL_Delay(10);
      cnt_press++;
      if (cnt_press == (2 * 100)) { poweroff(); }
    }
    if (cnt_press > 8) {
      ignition = !ignition;
      out_enable(IGNITION, ignition);
      beepShort(5);
    }
  }
}

#define PULL_EFFECTIVE_DELAY 4096
uint8_t detect_with_pull(GPIO_TypeDef* GPIOx, uint16_t GPIO_Pin, uint32_t mode) {
  GPIO_InitTypeDef GPIO_InitStruct;
  GPIO_InitStruct.Pin = GPIO_Pin;
  GPIO_InitStruct.Mode  = GPIO_MODE_INPUT;
  GPIO_InitStruct.Pull  = mode;
  HAL_GPIO_Init(GPIOx, &GPIO_InitStruct);
  for (volatile int i=0; i<PULL_EFFECTIVE_DELAY; i++);
  bool ret = HAL_GPIO_ReadPin(GPIOx, GPIO_Pin);
  GPIO_InitStruct.Pull  = GPIO_NOPULL;
  HAL_GPIO_Init(GPIOx, &GPIO_InitStruct);
  return ret;
}

uint8_t board_id(void) {
  return  (((!detect_with_pull(KEY1_PORT, KEY1_PIN, GPIO_PULLUP))) | (!detect_with_pull(KEY2_PORT, KEY2_PIN, GPIO_PULLUP)) << 1U);
}

uint8_t crc_checksum(uint8_t *dat, int len, const uint8_t poly) {
  uint8_t crc = 0xFFU;
  int i;
  int j;
  for (i = len - 1; i >= 0; i--) {
    crc ^= dat[i];
    for (j = 0; j < 8; j++) {
      if ((crc & 0x80U) != 0U) {
        crc = (uint8_t)((crc << 1) ^ poly);
      }
      else {
        crc <<= 1;
      }
    }
  }
  return crc;
}

/* =========================== Filtering Functions =========================== */

  /* Low pass filter fixed-point 32 bits: fixdt(1,32,16)
  * Max:  32767.99998474121
  * Min: -32768
  * Res:  1.52587890625e-05
  *
  * Inputs:       u     = int16 or int32
  * Outputs:      y     = fixdt(1,32,16)
  * Parameters:   coef  = fixdt(0,16,16) = [0,65535U]
  *
  * Example:
  * If coef = 0.8 (in floating point), then coef = 0.8 * 2^16 = 52429 (in fixed-point)
  * filtLowPass16(u, 52429, &y);
  * yint = (int16_t)(y >> 16); // the integer output is the fixed-point ouput shifted by 16 bits
  */
void filtLowPass32(int32_t u, uint16_t coef, int32_t *y) {
  int64_t tmp;
  tmp = ((int64_t)((u << 4) - (*y >> 12)) * coef) >> 4;
  tmp = CLAMP(tmp, -2147483648LL, 2147483647LL);  // Overflow protection: 2147483647LL = 2^31 - 1
  *y = (int32_t)tmp + (*y);
}

  /* rateLimiter16(int16_t u, int16_t rate, int16_t *y);
  * Inputs:       u     = int16
  * Outputs:      y     = fixdt(1,16,4)
  * Parameters:   rate  = fixdt(1,16,4) = [0, 32767] Do NOT make rate negative (>32767)
  */
void rateLimiter16(int16_t u, int16_t rate, int16_t *y) {
  int16_t q0;
  int16_t q1;

  q0 = (u << 4)  - *y;

  if (q0 > rate) {
    q0 = rate;
  } else {
    q1 = -rate;
    if (q0 < q1) {
      q0 = q1;
    }
  }

  *y = q0 + *y;
}
