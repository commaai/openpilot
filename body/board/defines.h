// Define to prevent recursive inclusion
#ifndef DEFINES_H
#define DEFINES_H

#include "stm32f4xx_hal.h"
#include "config.h"

#define LEFT_TIM TIM8
#define LEFT_TIM_U CCR1
#define LEFT_TIM_UH_PIN GPIO_PIN_6
#define LEFT_TIM_UH_PORT GPIOC
#define LEFT_TIM_UL_PIN GPIO_PIN_7
#define LEFT_TIM_UL_PORT GPIOA
#define LEFT_TIM_V CCR2
#define LEFT_TIM_VH_PIN GPIO_PIN_7
#define LEFT_TIM_VH_PORT GPIOC
#define LEFT_TIM_VL_PIN GPIO_PIN_0
#define LEFT_TIM_VL_PORT GPIOB
#define LEFT_TIM_W CCR3
#define LEFT_TIM_WH_PIN GPIO_PIN_8
#define LEFT_TIM_WH_PORT GPIOC
#define LEFT_TIM_WL_PIN GPIO_PIN_1
#define LEFT_TIM_WL_PORT GPIOB

#define RIGHT_TIM TIM1
#define RIGHT_TIM_U CCR1
#define RIGHT_TIM_UH_PIN GPIO_PIN_8
#define RIGHT_TIM_UH_PORT GPIOA
#define RIGHT_TIM_UL_PIN GPIO_PIN_13
#define RIGHT_TIM_UL_PORT GPIOB
#define RIGHT_TIM_V CCR2
#define RIGHT_TIM_VH_PIN GPIO_PIN_9
#define RIGHT_TIM_VH_PORT GPIOA
#define RIGHT_TIM_VL_PIN GPIO_PIN_14
#define RIGHT_TIM_VL_PORT GPIOB
#define RIGHT_TIM_W CCR3
#define RIGHT_TIM_WH_PIN GPIO_PIN_10
#define RIGHT_TIM_WH_PORT GPIOA
#define RIGHT_TIM_WL_PIN GPIO_PIN_15
#define RIGHT_TIM_WL_PORT GPIOB

#define LEFT_DC_CUR_PIN GPIO_PIN_3
#define LEFT_U_CUR_PIN GPIO_PIN_5
#define LEFT_V_CUR_PIN GPIO_PIN_6

#define LEFT_DC_CUR_PORT GPIOA
#define LEFT_U_CUR_PORT GPIOA
#define LEFT_V_CUR_PORT GPIOA

#define RIGHT_DC_CUR_PIN GPIO_PIN_2
#define RIGHT_U_CUR_PIN GPIO_PIN_0
#define RIGHT_V_CUR_PIN GPIO_PIN_1

#define RIGHT_DC_CUR_PORT GPIOA
#define RIGHT_U_CUR_PORT GPIOA
#define RIGHT_V_CUR_PORT GPIOA

#define BATT_PIN GPIO_PIN_4
#define BATT_PORT GPIOA

#define BUZZER_PIN GPIO_PIN_2
#define BUZZER_PORT GPIOC

#define OFF_PIN GPIO_PIN_4
#define OFF_PORT GPIOC

#define BUTTON_PIN GPIO_PIN_8
#define BUTTON_PORT GPIOB

#define CHARGER_PIN GPIO_PIN_12
#define CHARGER_PORT GPIOA

#define SW_I2C1_SCL_GPIO  GPIOB
#define SW_I2C1_SDA_GPIO  GPIOB
#define SW_I2C1_SCL_PIN   GPIO_PIN_3
#define SW_I2C1_SDA_PIN   GPIO_PIN_4

// UID pins
#define KEY1_PIN GPIO_PIN_10
#define KEY1_PORT GPIOB
#define KEY2_PIN GPIO_PIN_9
#define KEY2_PORT GPIOC

#define DELAY_TIM_FREQUENCY_US 1000000

#define MILLI_R (R * 1000)
#define MILLI_PSI (PSI * 1000)
#define MILLI_V (V * 1000)

#define NO 0
#define YES 1
#define ABS(a) (((a) < 0) ? -(a) : (a))
#define LIMIT(x, lowhigh) (((x) > (lowhigh)) ? (lowhigh) : (((x) < (-lowhigh)) ? (-lowhigh) : (x)))
#define SAT(x, lowhigh) (((x) > (lowhigh)) ? (1.0f) : (((x) < (-lowhigh)) ? (-1.0f) : (0.0f)))
#define SAT2(x, low, high) (((x) > (high)) ? (1.0f) : (((x) < (low)) ? (-1.0f) : (0.0f)))
#define STEP(from, to, step) (((from) < (to)) ? (MIN((from) + (step), (to))) : (MAX((from) - (step), (to))))
#define DEG(a) ((a)*M_PI / 180.0f)
#define RAD(a) ((a)*180.0f / M_PI)
#define SIGN(a) (((a) < 0) ? (-1) : (((a) > 0) ? (1) : (0)))
#define CLAMP(x, low, high) (((x) > (high)) ? (high) : (((x) < (low)) ? (low) : (x)))
#define IN_RANGE(x, low, high) (((x) >= (low)) && ((x) <= (high)))
#define SCALE(value, high, max) MIN(MAX(((max) - (value)) / ((max) - (high)), 0.0f), 1.0f)
#define MIN(a, b) (((a) < (b)) ? (a) : (b))
#define MAX(a, b) (((a) > (b)) ? (a) : (b))
#define MIN3(a, b, c) MIN(a, MIN(b, c))
#define MAX3(a, b, c) MAX(a, MAX(b, c))
#define ARRAY_LEN(x) (uint32_t)(sizeof(x) / sizeof(*(x)))
#define MAP(x, in_min, in_max, out_min, out_max) (((((x) - (in_min)) * ((out_max) - (out_min))) / ((in_max) - (in_min))) + (out_min))

#define GET_MAILBOX_BYTE(msg, b) (((int)(b) > 3) ? (((msg)->RDHR >> (8U * ((unsigned int)(b) % 4U))) & 0xFFU) : (((msg)->RDLR >> (8U * (unsigned int)(b))) & 0xFFU))
#define GET_MAILBOX_BYTES_04(msg) ((msg)->RDLR)
#define GET_MAILBOX_BYTES_48(msg) ((msg)->RDHR)

#define BOOT_NORMAL 0xdeadb111U
#define ENTER_SOFTLOADER_MAGIC 0xdeadc0deU

#define APP_START_ADDRESS 0x8004000U
#define DEVICE_SERIAL_NUMBER_ADDRESS 0x1FFF79C0U

#define COMPILE_TIME_ASSERT(pred) ((void)sizeof(char[1 - (2 * ((int)(!(pred))))]))

#define LED_RED     0
#define LED_GREEN   1
#define LED_BLUE    2
#define IGNITION    3
#define POWERSWITCH 4
#define TRANSCEIVER 5

#define HW_TYPE_BASE 0
#define HW_TYPE_KNEE 3

typedef struct {
  uint32_t rrB;
  uint32_t rrC;
  uint32_t rlA;
  uint32_t rlB;
  uint32_t dcr;
  uint32_t dcl;
  uint32_t batt1;
  uint32_t temp;
} adc_buf_t;

typedef struct {
  GPIO_TypeDef* hall_portA;
  uint16_t hall_pinA;
  GPIO_TypeDef* hall_portB;
  uint16_t hall_pinB;
  GPIO_TypeDef* hall_portC;
  uint16_t hall_pinC;
} hall_sensor;

typedef struct {
  hall_sensor hall_left;
  hall_sensor hall_right;

  CAN_TypeDef* CAN;
  uint8_t can_alt_tx;
  uint8_t can_alt_rx;
  GPIO_TypeDef* can_portTX;
  uint16_t can_pinTX;
  GPIO_TypeDef* can_portRX;
  uint16_t can_pinRX;
  GPIO_TypeDef* can_portEN;
  uint16_t can_pinEN;

  GPIO_TypeDef* ignition_port;
  uint16_t ignition_pin;

  uint32_t can_addr_offset;

  GPIO_TypeDef* led_portR;
  uint16_t led_pinR;
  GPIO_TypeDef* led_portG;
  uint16_t led_pinG;
  GPIO_TypeDef* led_portB;
  uint16_t led_pinB;

} board_t;

#endif // DEFINES_H
