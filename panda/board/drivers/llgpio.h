#define MODE_INPUT 0
#define MODE_OUTPUT 1
#define MODE_ALTERNATE 2
#define MODE_ANALOG 3

#define PULL_NONE 0
#define PULL_UP 1
#define PULL_DOWN 2

void set_gpio_mode(GPIO_TypeDef *GPIO, int pin, int mode) {
  uint32_t tmp = GPIO->MODER;
  tmp &= ~(3 << (pin*2));
  tmp |= (mode << (pin*2));
  GPIO->MODER = tmp;
}

void set_gpio_output(GPIO_TypeDef *GPIO, int pin, bool enabled) {
  if (enabled) {
    GPIO->ODR |= (1 << pin);
  } else {
    GPIO->ODR &= ~(1 << pin);
  }
  set_gpio_mode(GPIO, pin, MODE_OUTPUT);
}

void set_gpio_alternate(GPIO_TypeDef *GPIO, int pin, int mode) {
  uint32_t tmp = GPIO->AFR[pin>>3];
  tmp &= ~(0xF << ((pin&7)*4));
  tmp |= mode << ((pin&7)*4);
  GPIO->AFR[pin>>3] = tmp;
  set_gpio_mode(GPIO, pin, MODE_ALTERNATE);
}

void set_gpio_pullup(GPIO_TypeDef *GPIO, int pin, int mode) {
  uint32_t tmp = GPIO->PUPDR;
  tmp &= ~(3 << (pin*2));
  tmp |= (mode << (pin*2));
  GPIO->PUPDR = tmp;
}

int get_gpio_input(GPIO_TypeDef *GPIO, int pin) {
  return (GPIO->IDR & (1U << pin)) == (1U << pin);
}

