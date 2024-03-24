/*
HSE: 25MHz
PLL1Q: 80MHz (for FDCAN)
HSI48 enabled (for USB)
CPU: 240MHz
CPU Systick: 240MHz
AXI: 120MHz
HCLK3: 60MHz
APB3 per: 60MHz
AHB1,2 per: 120MHz
APB1 per: 60MHz
APB1 tim: 120MHz
APB2 per: 60MHz
APB2 tim: 120MHz
AHB4 per: 120MHz
APB4 per: 60MHz
PCLK1: 60MHz (for USART2,3,4,5,7,8)
*/

void clock_init(void) {
  // Set power mode to direct SMPS power supply(depends on the board layout)
#ifndef STM32H723
  register_set(&(PWR->CR3), PWR_CR3_SMPSEN, 0xFU); // powered only by SMPS
#else
  register_set(&(PWR->CR3), PWR_CR3_LDOEN, 0xFU);
#endif
  // Set VOS level (VOS3 to 170Mhz, VOS2 to 300Mhz, VOS1 to 400Mhz, VOS0 to 550Mhz)
  register_set(&(PWR->D3CR), PWR_D3CR_VOS_1 | PWR_D3CR_VOS_0, 0xC000U); //VOS1, needed for 80Mhz CAN FD
  while ((PWR->CSR1 & PWR_CSR1_ACTVOSRDY) == 0);
  while ((PWR->CSR1 & PWR_CSR1_ACTVOS) != (PWR->D3CR & PWR_D3CR_VOS)); // check that VOS level was actually set

  // Configure Flash ACR register LATENCY and WRHIGHFREQ (VOS0 range!)
  register_set(&(FLASH->ACR), FLASH_ACR_LATENCY_2WS | 0x20U, 0x3FU); // VOS2, AXI 100MHz-150MHz
  // enable external oscillator HSE
  register_set_bits(&(RCC->CR), RCC_CR_HSEON);
  while ((RCC->CR & RCC_CR_HSERDY) == 0);
  // enable internal HSI48 for USB FS kernel
  register_set_bits(&(RCC->CR), RCC_CR_HSI48ON);
  while ((RCC->CR & RCC_CR_HSI48RDY) == 0);
  // Specify the frequency source for PLL1, divider for DIVM1, DIVM2, DIVM3 : HSE, 5, 5, 5
  register_set(&(RCC->PLLCKSELR), RCC_PLLCKSELR_PLLSRC_HSE | RCC_PLLCKSELR_DIVM1_0 | RCC_PLLCKSELR_DIVM1_2 | RCC_PLLCKSELR_DIVM2_0 | RCC_PLLCKSELR_DIVM2_2 | RCC_PLLCKSELR_DIVM3_0 | RCC_PLLCKSELR_DIVM3_2, 0x3F3F3F3U);

  // *** PLL1 start ***
  // Specify multiplier N and dividers P, Q, R for PLL1 : 48, 1, 3, 2 (clock 240Mhz, PLL1Q 80Mhz for CAN FD)
  register_set(&(RCC->PLL1DIVR), 0x102002FU, 0x7F7FFFFFU);
  // Specify the input and output frequency ranges, enable dividers for PLL1
  register_set(&(RCC->PLLCFGR), RCC_PLLCFGR_PLL1RGE_2 | RCC_PLLCFGR_DIVP1EN | RCC_PLLCFGR_DIVQ1EN | RCC_PLLCFGR_DIVR1EN, 0x7000CU);
  // Enable PLL1
  register_set_bits(&(RCC->CR), RCC_CR_PLL1ON);
  while((RCC->CR & RCC_CR_PLL1RDY) == 0);
  // *** PLL1 end ***

  //////////////OTHER CLOCKS////////////////////
  // RCC HCLK Clock Source / RCC APB3 Clock Source / RCC SYS Clock Source
  register_set(&(RCC->D1CFGR), RCC_D1CFGR_HPRE_DIV2 | RCC_D1CFGR_D1PPRE_DIV2 | RCC_D1CFGR_D1CPRE_DIV1, 0xF7FU);
  // RCC APB1 Clock Source / RCC APB2 Clock Source
  register_set(&(RCC->D2CFGR), RCC_D2CFGR_D2PPRE1_DIV2 | RCC_D2CFGR_D2PPRE2_DIV2, 0x770U);
  // RCC APB4 Clock Source
  register_set(&(RCC->D3CFGR), RCC_D3CFGR_D3PPRE_DIV2, 0x70U);

  // Set SysClock source to PLL
  register_set(&(RCC->CFGR), RCC_CFGR_SW_PLL1, 0x7U);
  while((RCC->CFGR & RCC_CFGR_SWS) != RCC_CFGR_SWS_PLL1);
  //////////////END OTHER CLOCKS////////////////////

  // Configure clock source for USB (HSI48)
  register_set(&(RCC->D2CCIP2R), RCC_D2CCIP2R_USBSEL_1 | RCC_D2CCIP2R_USBSEL_0, RCC_D2CCIP2R_USBSEL);
  // Configure clock source for FDCAN (PLL1Q at 80Mhz)
  register_set(&(RCC->D2CCIP1R), RCC_D2CCIP1R_FDCANSEL_0, RCC_D2CCIP1R_FDCANSEL);
  // Configure clock source for ADC1,2,3 (per_ck(currently HSE))
  register_set(&(RCC->D3CCIPR), RCC_D3CCIPR_ADCSEL_1, RCC_D3CCIPR_ADCSEL);
  //Enable the Clock Security System
  register_set_bits(&(RCC->CR), RCC_CR_CSSHSEON);
  //Enable Vdd33usb supply level detector
  register_set_bits(&(PWR->CR3), PWR_CR3_USB33DEN);
}
