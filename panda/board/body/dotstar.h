#pragma once

#include <stdbool.h>
#include <stdint.h>

#include "board/config.h"
#include "board/body/boards/board_declarations.h"

#define DOTSTAR_LED_COUNT 10U
#define DOTSTAR_GLOBAL_BRIGHTNESS_MAX 31U

typedef struct {
  uint8_t r;
  uint8_t g;
  uint8_t b;
} dotstar_rgb_t;

typedef struct {
  bool initialized;
  uint8_t global_brightness;
  dotstar_rgb_t pixels[DOTSTAR_LED_COUNT];
} dotstar_state_t;

static dotstar_state_t dotstar_state = {
  .initialized = false,
  .global_brightness = DOTSTAR_GLOBAL_BRIGHTNESS_MAX,
};

static inline void dotstar_set_clk(bool high) {
  if (high) {
    DOTSTAR_CLK_PORT->BSRR = (uint32_t)(1U << DOTSTAR_CLK_PIN);
  } else {
    DOTSTAR_CLK_PORT->BSRR = (uint32_t)(1U << (DOTSTAR_CLK_PIN + 16U));
  }
}

static inline void dotstar_set_data(bool high) {
  if (high) {
    DOTSTAR_DATA_PORT->BSRR = (uint32_t)(1U << DOTSTAR_DATA_PIN);
  } else {
    DOTSTAR_DATA_PORT->BSRR = (uint32_t)(1U << (DOTSTAR_DATA_PIN + 16U));
  }
}

static inline void dotstar_write_byte(uint8_t value) {
  for (int8_t bit = 7; bit >= 0; bit--) {
    dotstar_set_data(((value >> bit) & 0x1U) != 0U);
    delay(15);
    dotstar_set_clk(true);
    delay(15);
    dotstar_set_clk(false);
    delay(15);
  }
}

static inline uint16_t dotstar_latch_len(uint16_t led_count) {
  uint16_t len = (uint16_t)((led_count + 15U) / 16U);
  if (len < led_count) {
    len = led_count;
  }
  return (len < 4U) ? 4U : len;
}

static inline void dotstar_send_start_frame(void) {
  for (uint8_t i = 0U; i < 4U; i++) {
    dotstar_write_byte(0x00U);
  }
}

static inline void dotstar_send_end_frame(uint16_t led_count) {
  uint16_t latch_len = dotstar_latch_len(led_count);
  for (uint16_t i = 0U; i < latch_len; i++) {
    dotstar_write_byte(0xFFU);
  }
}

static inline void dotstar_show(void) {
  if (!dotstar_state.initialized) {
    return;
  }

  dotstar_send_start_frame();

  uint8_t prefix = (uint8_t)(0xE0U | dotstar_state.global_brightness);
  for (uint16_t i = 0U; i < DOTSTAR_LED_COUNT; i++) {
    dotstar_rgb_t *pixel = &dotstar_state.pixels[i];
    dotstar_write_byte(prefix);
    dotstar_write_byte(pixel->b);
    dotstar_write_byte(pixel->g);
    dotstar_write_byte(pixel->r);
  }

  dotstar_send_end_frame(DOTSTAR_LED_COUNT);
}

static inline void dotstar_init(void) {
  set_gpio_pullup(DOTSTAR_CLK_PORT, DOTSTAR_CLK_PIN, PULL_NONE);
  set_gpio_output_type(DOTSTAR_CLK_PORT, DOTSTAR_CLK_PIN, OUTPUT_TYPE_PUSH_PULL);
  set_gpio_mode(DOTSTAR_CLK_PORT, DOTSTAR_CLK_PIN, MODE_OUTPUT);
  register_set(&(DOTSTAR_CLK_PORT->OSPEEDR), GPIO_OSPEEDR_OSPEED5, GPIO_OSPEEDR_OSPEED5_Msk);
  dotstar_set_clk(false);

  set_gpio_pullup(DOTSTAR_DATA_PORT, DOTSTAR_DATA_PIN, PULL_NONE);
  set_gpio_output_type(DOTSTAR_DATA_PORT, DOTSTAR_DATA_PIN, OUTPUT_TYPE_PUSH_PULL);
  set_gpio_mode(DOTSTAR_DATA_PORT, DOTSTAR_DATA_PIN, MODE_OUTPUT);
  register_set(&(DOTSTAR_DATA_PORT->OSPEEDR), GPIO_OSPEEDR_OSPEED5, GPIO_OSPEEDR_OSPEED5_Msk);
  dotstar_set_data(false);

  dotstar_state.initialized = true;
  dotstar_state.global_brightness = DOTSTAR_GLOBAL_BRIGHTNESS_MAX;

  for (uint16_t i = 0U; i < DOTSTAR_LED_COUNT; i++) {
    dotstar_state.pixels[i].r = 0U;
    dotstar_state.pixels[i].g = 0U;
    dotstar_state.pixels[i].b = 0U;
  }

  dotstar_show();
}

static inline void dotstar_set_pixel(uint16_t index, uint8_t r, uint8_t g, uint8_t b) {
  if (!dotstar_state.initialized || (index >= DOTSTAR_LED_COUNT)) {
    return;
  }
  dotstar_state.pixels[index].r = r;
  dotstar_state.pixels[index].g = g;
  dotstar_state.pixels[index].b = b;
}

static inline void dotstar_fill(uint8_t r, uint8_t g, uint8_t b) {
  if (!dotstar_state.initialized) {
    return;
  }
  for (uint16_t i = 0U; i < DOTSTAR_LED_COUNT; i++) {
    dotstar_state.pixels[i].r = r;
    dotstar_state.pixels[i].g = g;
    dotstar_state.pixels[i].b = b;
  }
}

static inline void dotstar_set_global_brightness(uint8_t brightness) {
  dotstar_state.global_brightness = (brightness > DOTSTAR_GLOBAL_BRIGHTNESS_MAX) ? DOTSTAR_GLOBAL_BRIGHTNESS_MAX : brightness;
}

static inline void dotstar_hue_to_rgb(uint16_t hue, uint8_t *r, uint8_t *g, uint8_t *b) {
  hue %= 765U;
  if (hue < 255U) {
    *r = (uint8_t)(255U - hue);
    *g = (uint8_t)hue;
    *b = 0U;
  } else if (hue < 510U) {
    hue -= 255U;
    *r = 0U;
    *g = (uint8_t)(255U - hue);
    *b = (uint8_t)hue;
  } else {
    hue -= 510U;
    *r = (uint8_t)hue;
    *g = 0U;
    *b = (uint8_t)(255U - hue);
  }
}

static inline void dotstar_run_rainbow(uint32_t now_us) {
  uint32_t brightness_phase = (now_us / 40000U) % 62U;
  uint8_t brightness = (brightness_phase <= 31U) ? (uint8_t)(brightness_phase + 1U) : (uint8_t)(62U - brightness_phase);
  if (brightness == 0U) {
    brightness = 1U;
  }
  dotstar_set_global_brightness(brightness);

  uint32_t base_hue = (now_us / 10000U) % 765U;
  for (uint16_t i = 0U; i < DOTSTAR_LED_COUNT; i++) {
    uint16_t hue = (uint16_t)((base_hue + (i * 70U)) % 765U);
    uint8_t r, g, b;
    dotstar_hue_to_rgb(hue, &r, &g, &b);
    dotstar_set_pixel(i, r, g, b);
  }
}

static inline void dotstar_apply_breathe(dotstar_rgb_t color, uint32_t now_us, uint32_t cycle_us) {
  if (!dotstar_state.initialized) {
    return;
  }

  if (cycle_us == 0U) {
    dotstar_set_global_brightness(DOTSTAR_GLOBAL_BRIGHTNESS_MAX);
    dotstar_fill(color.r, color.g, color.b);
    return;
  }

  uint32_t phase = now_us % cycle_us;
  uint32_t half_cycle = cycle_us / 2U;
  if (half_cycle == 0U) {
    half_cycle = 1U;
  }

  uint32_t amplitude = (phase <= half_cycle) ? phase : (cycle_us - phase);
  uint32_t scale = (amplitude * 255U) / half_cycle;
  if (scale > 255U) {
    scale = 255U;
  }

  uint8_t r = (uint8_t)((color.r * scale) / 255U);
  uint8_t g = (uint8_t)((color.g * scale) / 255U);
  uint8_t b = (uint8_t)((color.b * scale) / 255U);

  dotstar_set_global_brightness(DOTSTAR_GLOBAL_BRIGHTNESS_MAX);
  dotstar_fill(r, g, b);
}